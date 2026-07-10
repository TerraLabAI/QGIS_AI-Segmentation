"""Automatic post-run review panel construction.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out of auto_build.py so agents and humans work on one concern per
file. Methods are plain mixin members: widgets/signals live on the dock
instance.
"""
from __future__ import annotations


from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


from ...core.i18n import tr
from ...core.review_defaults import (
    AUTO_REVIEW_CLEAN_DEFAULT as _AUTO_REVIEW_CLEAN_DEFAULT,
    AUTO_REVIEW_EXPAND_DEFAULT as _AUTO_REVIEW_EXPAND_DEFAULT,
    AUTO_REVIEW_FILL_HOLES_DEFAULT as _AUTO_REVIEW_FILL_HOLES_DEFAULT,
    AUTO_REVIEW_ORTHO_DEFAULT as _AUTO_REVIEW_ORTHO_DEFAULT,
    AUTO_REVIEW_SIMPLIFY_DEFAULT as _AUTO_REVIEW_SIMPLIFY_DEFAULT,
    AUTO_REVIEW_SMOOTH_DEFAULT as _AUTO_REVIEW_SMOOTH_DEFAULT,
)
from .guidance import (
    BLUE_TINT,
    HINT_REVIEW_CONFIDENCE,
    DismissibleHint,
)
from .styles import (
    _BTN_GHOST,
    _BTN_GREEN,
    _CARD_QSS,
    _REVIEW_CONF_MAX,
    _REVIEW_CONF_MIN,
    _REVIEW_CONF_SPIN_MIN,
    _REVIEW_CONF_STEP,
    _SLIDER_QSS,
)

# Outline-blue secondary button, review-only: blue is the plugin's
# "temporary / still editing" colour, so "Refine in Manual mode" reads as
# "keep working on this result" next to the green "done" Export. Local until
# another panel needs it (then it moves to styles.py).
_BTN_OUTLINE_BLUE = (
    "QPushButton { background-color: transparent; color: #1e88e5;"
    " border: 1px solid #1e88e5; border-radius: 4px; font-weight: 600;"
    " padding: 6px 12px; }"
    "QPushButton:hover { background-color: rgba(30,136,229,0.12); }"
    "QPushButton:disabled { color: #666666; border-color: #666666; }"
)


def _make_review_group_label(text: str) -> QLabel:
    """Tiny uppercase section label for the review action groups ("Keep this
    result" / "Start over"): same typography as the collapsible-section
    header, minus the chevron and border. The two labels carry the button
    hierarchy: each action lives under the OUTCOME it produces."""
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        " color: rgba(128,128,128,0.95); padding-top: 6px;")
    return lbl


class DockAutoReviewBuildMixin:
    """Automatic post-run review panel construction."""

    def _setup_auto_review_panel(self, parent_layout):
        """Post-run review panel (hidden until a run finishes successfully)."""
        self.auto_review_panel = QWidget()
        self.auto_review_panel.setVisible(False)
        _review_layout = QVBoxLayout(self.auto_review_panel)
        _review_layout.setContentsMargins(0, 0, 0, 0)
        # 8px between the three sibling blocks (colors / confidence / refine)
        # keeps a calm, clearly hierarchised review area.
        _review_layout.setSpacing(8)

        # 1. SUMMARY line: the review header, above every box. ONE compact line
        # ("✓ 158 of 352 shown · 194 below 65%", built by
        # _format_auto_review_count) naming what the confidence filter keeps.
        self._auto_review_count_label = QLabel("")
        self._auto_review_count_label.setWordWrap(True)
        self._auto_review_count_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: palette(text);"
            " padding-bottom: 2px;")
        _review_layout.addWidget(self._auto_review_count_label)

        # Inline install banner (D1): when Refine in Manual mode is clicked
        # without the local AI installed, the setup runs in the BACKGROUND while
        # this review stays fully usable. This banner shows the progress; the
        # Refine button re-enables and the handoff opens automatically once the AI
        # is ready (driven by set_auto_review_installing + set_install_progress).
        self._auto_review_installing = False
        self.auto_review_install_banner = QWidget()
        self.auto_review_install_banner.setObjectName("autoReviewInstallBanner")
        self.auto_review_install_banner.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_review_install_banner.setStyleSheet(
            "QWidget#autoReviewInstallBanner {"
            " background-color: rgba(30, 136, 229, 0.10);"
            " border: 1px solid rgba(30, 136, 229, 0.35);"
            " border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }")
        _install_col = QVBoxLayout(self.auto_review_install_banner)
        _install_col.setContentsMargins(10, 8, 10, 8)
        _install_col.setSpacing(4)
        self.auto_review_install_label = QLabel(
            tr("Setting up Manual mode in the background..."))
        self.auto_review_install_label.setWordWrap(True)
        self.auto_review_install_label.setStyleSheet(
            "font-size: 11px; color: palette(text);")
        _install_col.addWidget(self.auto_review_install_label)
        self.auto_review_install_progress = QProgressBar()
        self.auto_review_install_progress.setRange(0, 100)
        self.auto_review_install_progress.setValue(0)
        self.auto_review_install_progress.setTextVisible(False)
        self.auto_review_install_progress.setMaximumHeight(6)
        _install_col.addWidget(self.auto_review_install_progress)
        self.auto_review_install_banner.setVisible(False)
        _review_layout.addWidget(self.auto_review_install_banner)

        # 2. DISPLAY COLORS box: its own compact framed card. How the detections
        # are drawn during review (visual only, never geometry/filters/export).
        _colors_box = QWidget()
        _colors_box.setObjectName("autoColorsBox")
        _colors_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        _colors_box.setStyleSheet(
            _CARD_QSS.format(name="autoColorsBox") + "QLabel { background: transparent; border: none; }"
        )
        _colors_layout = QVBoxLayout(_colors_box)
        _colors_layout.setContentsMargins(10, 10, 10, 10)
        _colors_layout.setSpacing(6)

        # Display colours sit at the TOP of the review card (they are the first
        # thing a reviewer reads on the map): how the detections are drawn during
        # review. Normal = plain outline; Confidence = heatmap on the per-object
        # score (so the confidence slider shows what it cuts); Random = one
        # distinct colour per object (tell touching or merged objects apart).
        # Purely visual: it changes the renderer only, never the geometry, the
        # filters or what gets exported.
        _display_row = QHBoxLayout()
        _display_lbl = QLabel(tr("Display colors:"))
        _display_lbl.setStyleSheet("font-size: 11px;")
        _display_row.addWidget(_display_lbl)
        _display_row.addStretch()
        self.auto_display_combo = QComboBox()
        self.auto_display_combo.addItem(tr("Normal"), "normal")
        self.auto_display_combo.addItem(tr("Outline"), "outline")
        self.auto_display_combo.addItem(tr("Confidence"), "confidence")
        self.auto_display_combo.addItem(tr("Random"), "random")
        # Random by default: a fresh review opens with one colour per object so
        # instances read as distinct (re-seeded per review by the plugin).
        self.auto_display_combo.setCurrentIndex(
            max(0, self.auto_display_combo.findData("random")))
        self.auto_display_combo.setToolTip(tr(
            "How detections are coloured on the map (visual only): Normal outline, "
            "Confidence heatmap (green sure, red uncertain), or a random colour per "
            "object to tell them apart."))
        self.auto_display_combo.currentIndexChanged.connect(
            lambda _i: self.auto_display_mode_changed.emit(
                self.auto_display_combo.currentData() or "confidence"))
        _display_row.addWidget(self.auto_display_combo)
        _colors_layout.addLayout(_display_row)

        # One muted line under the combo spelling out what the colours MEAN for
        # the selected mode (set per mode from _on_auto_display_mode_changed).
        # Seeded with the Random-mode text since the combo starts on Random.
        self.auto_display_legend = QLabel(
            tr("One color per object - check neighbors are separated"))
        self.auto_display_legend.setWordWrap(True)
        self.auto_display_legend.setStyleSheet(
            "font-size: 11px; color: rgba(128,128,128,0.95);")
        _colors_layout.addWidget(self.auto_display_legend)
        _review_layout.addWidget(_colors_box)

        # 3. CONFIDENCE box: its own framed card holding the header, histogram,
        # slider, end labels and (at the bottom, as a quiet last line) the
        # dismissible confidence tip that used to float above the whole panel.
        _conf_box = QWidget()
        _conf_box.setObjectName("autoConfBox")
        _conf_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        _conf_box.setStyleSheet(
            _CARD_QSS.format(name="autoConfBox") + "QLabel { background: transparent; border: none; }" + _SLIDER_QSS
        )
        _conf_layout = QVBoxLayout(_conf_box)
        _conf_layout.setContentsMargins(10, 10, 10, 10)
        _conf_layout.setSpacing(6)

        # Confidence slider (live post-run re-filter): the run keeps every mask
        # above a low recall floor, so dragging this re-filters the detections
        # instantly with no re-detection and no extra credits. Fires on release
        # (not every tick) so the re-merge runs once per adjustment.
        _conf_hdr = QHBoxLayout()
        _conf_review_lbl = QLabel(tr("Confidence"))
        _conf_review_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
        _conf_hdr.addWidget(_conf_review_lbl)
        _conf_hdr.addStretch()
        # Editable number so the user can dial an exact cutoff instead of having
        # to nudge the slider. Slider and spinbox stay in sync; either one drives
        # the (debounced) re-filter.
        self.auto_review_confidence_spin = QSpinBox()
        self.auto_review_confidence_spin.setRange(_REVIEW_CONF_SPIN_MIN, _REVIEW_CONF_MAX)
        # Free 1% precision in the number box: the spinbox is the exact-cutoff
        # control; only the slider snaps to _REVIEW_CONF_STEP.
        self.auto_review_confidence_spin.setSingleStep(1)
        self.auto_review_confidence_spin.setValue(30)
        self.auto_review_confidence_spin.setSuffix("%")
        self.auto_review_confidence_spin.setMinimumWidth(62)
        self.auto_review_confidence_spin.setMaximumWidth(78)
        _conf_hdr.addWidget(self.auto_review_confidence_spin)
        _conf_layout.addLayout(_conf_hdr)
        # Score distribution strip above the slider: bars right of the cutoff are
        # bright (kept), left are dimmed (filtered out). Visual only.
        from ..confidence_histogram import ConfidenceHistogram
        self.auto_conf_histogram = ConfidenceHistogram()
        self.auto_conf_histogram.setToolTip(
            tr("How many objects sit at each confidence level."))
        _conf_layout.addWidget(self.auto_conf_histogram)
        self.auto_review_confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.auto_review_confidence_slider.setRange(_REVIEW_CONF_MIN, _REVIEW_CONF_MAX)
        self.auto_review_confidence_slider.setValue(30)
        self.auto_review_confidence_slider.setSingleStep(_REVIEW_CONF_STEP)
        self.auto_review_confidence_slider.setPageStep(_REVIEW_CONF_STEP)
        self.auto_review_confidence_slider.setMinimumHeight(26)
        self.auto_review_confidence_slider.setToolTip(tr(
            "Filter detections by confidence. Lower shows more (and noisier),"
            " higher keeps only the strongest. Free and instant."))
        # Dragging the slider mirrors into the spinbox live (cheap); the heavy
        # re-merge runs once via the debounce timer on release / spinbox edit.
        self.auto_review_confidence_slider.valueChanged.connect(self._on_conf_slider_moved)
        self.auto_review_confidence_slider.sliderReleased.connect(self._schedule_conf_refilter)
        self.auto_review_confidence_spin.valueChanged.connect(self._on_conf_spin_changed)
        _conf_layout.addWidget(self.auto_review_confidence_slider)

        # End labels so the slider direction reads at a glance (mirrors the
        # detail slider's Coarse/Fine ends).
        _conf_ends = QHBoxLayout()
        _conf_ends.setContentsMargins(2, 0, 2, 0)
        _conf_left = QLabel(tr("More objects"))
        _conf_left.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.95);")
        _conf_right = QLabel(tr("Only confident"))
        _conf_right.setStyleSheet("font-size: 10px; color: rgba(128,128,128,0.95);")
        _conf_ends.addWidget(_conf_left)
        _conf_ends.addStretch()
        _conf_ends.addWidget(_conf_right)
        _conf_layout.addLayout(_conf_ends)

        # Inline reason the slider is frozen after a Manual refine (a disabled
        # control with no caption reads as a bug). Hidden unless locked.
        self._auto_conf_lock_note = QLabel(
            "\U0001F512 " + tr("Locked - refined in Manual mode"))
        self._auto_conf_lock_note.setWordWrap(True)
        self._auto_conf_lock_note.setStyleSheet(
            "font-size: 10px; color: rgba(128,128,128,0.95);")
        self._auto_conf_lock_note.setVisible(False)
        _conf_layout.addWidget(self._auto_conf_lock_note)

        # Note shown when the review auto-lowered its starting cutoff because
        # nothing scored above 30%. Hidden by default; cleared on the first
        # user-initiated slider move.
        self.auto_conf_lowered_note = QLabel("")
        self.auto_conf_lowered_note.setWordWrap(True)
        self.auto_conf_lowered_note.setStyleSheet(
            "font-size: 10px; color: rgba(128,128,128,0.95);")
        self.auto_conf_lowered_note.setVisible(False)
        _conf_layout.addWidget(self.auto_conf_lowered_note)

        # Guidance tip: a one-line, dismissible explainer as the
        # quiet last line of the Confidence box, so a first-time reviewer knows
        # the confidence slider re-filters for free. Re-enable from Account
        # Settings.
        self.auto_review_confidence_hint = DismissibleHint(
            HINT_REVIEW_CONFIDENCE,
            tr("Tip: lower Confidence to reveal more detections, raise it to "
               "keep only the best."),
            tint=BLUE_TINT,
        )
        _conf_layout.addWidget(self.auto_review_confidence_hint)
        _review_layout.addWidget(_conf_box)

        # 4. REFINE DETECTIONS block: the collapsible toggle + its content are
        # their own sibling card below the Confidence box (no longer nested in
        # one big box). Collapsible "Refine detections" section (mirrors the
        # Manual panel's
        # collapsible "Refine selection" group): a full-width chevron header in
        # the design-system section-header typography toggles the shape + size
        # controls. Collapsed by default on every NEW review (reset via
        # set_auto_review_active) so Confidence, Refine-in-Manual and Export
        # stay above the fold; the toggle only flips visibility, so it never
        # emits a slider/spinbox signal.
        self._auto_shape_expanded = False
        # A full-width card-style header (not a bare underlined text row):
        # the section holds the run's auto-tuned shape controls, so it must
        # read as a real, clickable control at a glance (user feedback: the
        # old flat row was nearly invisible in the review panel).
        self.auto_shape_toggle_btn = QPushButton()
        self.auto_shape_toggle_btn.setStyleSheet(
            "QPushButton { font-size: 11px; color: palette(text);"
            " font-weight: bold; background-color: rgba(128, 128, 128, 0.10);"
            " border: 1px solid rgba(128, 128, 128, 0.30); border-radius: 4px;"
            " padding: 8px 10px; letter-spacing: 1px; text-align: left; }"
            "QPushButton:hover { color: #1e88e5;"
            " border-color: rgba(30, 136, 229, 0.7); }")
        self.auto_shape_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Never steal focus from the confidence spin / prompt on toggle.
        self.auto_shape_toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.auto_shape_toggle_btn.setToolTip(tr(
            "Optional shape and size controls: simplify outlines, clean edges, "
            "round corners, expand or shrink, fill holes, size filters."))
        self.auto_shape_toggle_btn.clicked.connect(self._on_auto_shape_toggle)
        self._refresh_auto_shape_header()
        _review_layout.addWidget(self.auto_shape_toggle_btn)

        # All shape + size controls live in one container so the header toggle
        # is a single setVisible (no per-row bookkeeping, no signals). Styled
        # as the same subtle bordered card as the Manual "Refine selection"
        # panel, with the shared visible checkbox indicators (the native
        # unchecked indicator can be invisible on dark themes).
        from .widgets import checkbox_indicator_qss
        self.auto_shape_content = QWidget()
        self.auto_shape_content.setObjectName("autoShapeContent")
        self.auto_shape_content.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.auto_shape_content.setStyleSheet(
            "QWidget#autoShapeContent {"
            " background-color: rgba(128, 128, 128, 0.08);"
            " border: 1px solid rgba(128, 128, 128, 0.2);"
            " border-radius: 4px; }"
            "QLabel { background: transparent; border: none; }" + checkbox_indicator_qss(self))
        _shape_layout = QVBoxLayout(self.auto_shape_content)
        _shape_layout.setContentsMargins(10, 6, 10, 10)
        _shape_layout.setSpacing(6)

        def _shape_section_label(text: str) -> QLabel:
            """Micro section header, same treatment as the Manual refine
            panel's OUTLINE / SELECTION headers (design-system section
            header: 10px bold uppercase, hairline underline)."""
            lbl = QLabel(text.upper())
            lbl.setStyleSheet(
                "font-size: 10px; color: palette(text); font-weight: bold; "
                "background: transparent; border: none; border-bottom: "
                "1px solid rgba(128, 128, 128, 0.35); "
                "padding: 8px 0px 4px 0px; margin-bottom: 4px; "
                "letter-spacing: 1px;")
            return lbl

        # Shape refinement (simplify outline / round corners / expand-shrink)
        # acts on the WHOLE detected objects, live and free. Simplify is a compact
        # spinbox and defaults low so the polygons keep their true detail out of
        # the box; the user opts in to smoothing by raising it. These mirror the
        # Manual refine knobs, applied to the merged geometry (there is no mask
        # left in review). Any change re-derives the visible set via
        # auto_refine_changed (debounced, cooperative).

        # Simplify: a compact numeric spinbox. 0 = faithful (no simplification),
        # higher = smoother outline. Drives the debounced re-derive directly.
        _simplify_hdr = QHBoxLayout()
        _simplify_lbl = QLabel(tr("Simplify outline:"))
        _simplify_lbl.setStyleSheet("font-size: 11px;")
        _simplify_lbl.setToolTip(tr(
            "Reduce small variations in the outline (0 = no change)"))
        _simplify_hdr.addWidget(_simplify_lbl)
        _simplify_hdr.addStretch()
        self.auto_simplify_spin = QDoubleSpinBox()
        self.auto_simplify_spin.setDecimals(1)
        self.auto_simplify_spin.setSingleStep(0.1)
        self.auto_simplify_spin.setRange(0.0, 1000.0)
        self.auto_simplify_spin.setValue(_AUTO_REVIEW_SIMPLIFY_DEFAULT)
        self.auto_simplify_spin.setSuffix(" px")
        self.auto_simplify_spin.setMinimumWidth(62)
        self.auto_simplify_spin.setMaximumWidth(78)
        self.auto_simplify_spin.setToolTip(_simplify_lbl.toolTip())
        _simplify_hdr.addWidget(self.auto_simplify_spin)

        # Clean edges: morphological opening (shrink-then-grow) that strips thin
        # attached fringe / tendrils the cloud model leaves around objects. Unlike Min size
        # (which only drops SEPARATE small features), this removes noise that is
        # part of the SAME polygon. Light default so panels/buildings come out
        # clean; 0 = off. px, converted to ground units by the source pixel size.
        _clean_hdr = QHBoxLayout()
        _clean_lbl = QLabel(tr("Clean edges:"))
        _clean_lbl.setStyleSheet("font-size: 11px;")
        _clean_lbl.setToolTip(tr(
            "Remove thin ragged fringe attached to the outline (0 = no change)"))
        _clean_hdr.addWidget(_clean_lbl)
        _clean_hdr.addStretch()
        self.auto_clean_spin = QDoubleSpinBox()
        self.auto_clean_spin.setDecimals(1)
        self.auto_clean_spin.setSingleStep(0.5)
        self.auto_clean_spin.setRange(0.0, 50.0)
        self.auto_clean_spin.setValue(_AUTO_REVIEW_CLEAN_DEFAULT)
        self.auto_clean_spin.setSuffix(" px")
        self.auto_clean_spin.setMinimumWidth(62)
        self.auto_clean_spin.setMaximumWidth(78)
        self.auto_clean_spin.setToolTip(_clean_lbl.toolTip())
        _clean_hdr.addWidget(self.auto_clean_spin)

        # Round corners checkbox + Expand/Contract spinbox, on compact rows to
        # keep the review panel short.
        _round_row = QHBoxLayout()
        _round_lbl = QLabel(tr("Round corners:"))
        _round_lbl.setStyleSheet("font-size: 11px;")
        _round_lbl.setToolTip(tr(
            "Round corners for natural shapes like trees and bushes. "
            "Increase 'Simplify outline' for smoother results."))
        self.auto_round_corners_check = QCheckBox()
        self.auto_round_corners_check.setChecked(_AUTO_REVIEW_SMOOTH_DEFAULT)
        self.auto_round_corners_check.setToolTip(_round_lbl.toolTip())
        _round_row.addWidget(_round_lbl)
        _round_row.addStretch()
        _round_row.addWidget(self.auto_round_corners_check)

        # Right angles: opt-in orthogonalization for man-made shapes. Applied to
        # ALL visible objects when checked (an explicit user intent beats an
        # automatic rectangularity gate, which would skip L/U-shaped buildings).
        _ortho_row = QHBoxLayout()
        _ortho_lbl = QLabel(tr("Right angles:"))
        _ortho_lbl.setStyleSheet("font-size: 11px;")
        _ortho_lbl.setToolTip(tr(
            "Snap edges to 90 degrees for man-made shapes like buildings, "
            "pools and solar panels."))
        self.auto_ortho_check = QCheckBox()
        self.auto_ortho_check.setChecked(_AUTO_REVIEW_ORTHO_DEFAULT)
        self.auto_ortho_check.setToolTip(_ortho_lbl.toolTip())
        _ortho_row.addWidget(_ortho_lbl)
        _ortho_row.addStretch()
        _ortho_row.addWidget(self.auto_ortho_check)

        _expand_row = QHBoxLayout()
        _expand_lbl = QLabel(tr("Expand/Contract:"))
        _expand_lbl.setStyleSheet("font-size: 11px;")
        _expand_lbl.setToolTip(tr("Positive = expand outward, Negative = shrink inward"))
        self.auto_expand_spin = QSpinBox()
        self.auto_expand_spin.setRange(-1000, 1000)
        self.auto_expand_spin.setValue(_AUTO_REVIEW_EXPAND_DEFAULT)
        self.auto_expand_spin.setSuffix(" px")
        self.auto_expand_spin.setMinimumWidth(62)
        self.auto_expand_spin.setMaximumWidth(78)
        _expand_row.addWidget(_expand_lbl)
        _expand_row.addStretch()
        _expand_row.addWidget(self.auto_expand_spin)

        # Fill holes: kept as an opt-in control but OFF by default so detected
        # holes (gaps between crowns, courtyards) are preserved unless asked.
        _fill_row = QHBoxLayout()
        _fill_lbl = QLabel(tr("Fill holes:"))
        _fill_lbl.setStyleSheet("font-size: 11px;")
        _fill_lbl.setToolTip(tr("Fill interior holes in the selection"))
        self.auto_fill_holes_check = QCheckBox()
        self.auto_fill_holes_check.setChecked(_AUTO_REVIEW_FILL_HOLES_DEFAULT)
        self.auto_fill_holes_check.setToolTip(_fill_lbl.toolTip())
        _fill_row.addWidget(_fill_lbl)
        _fill_row.addStretch()
        _fill_row.addWidget(self.auto_fill_holes_check)

        # Re-derive the visible set (via the debounced auto_refine_changed) on
        # any shape change; track the adjustment once per control per review.
        self.auto_simplify_spin.valueChanged.connect(
            lambda v: self._on_shape_control_changed("simplify", v))
        self.auto_clean_spin.valueChanged.connect(
            lambda v: self._on_shape_control_changed("clean", v))
        self.auto_round_corners_check.stateChanged.connect(
            lambda s: self._on_shape_control_changed("round_corners", s))
        self.auto_ortho_check.stateChanged.connect(
            lambda s: self._on_shape_control_changed("right_angles", s))
        self.auto_expand_spin.valueChanged.connect(
            lambda v: self._on_shape_control_changed("expand", v))
        self.auto_fill_holes_check.stateChanged.connect(
            lambda s: self._on_shape_control_changed("fill_holes", s))

        # Size filters: drop noise (below Min) and oversized blobs (above Max) by
        # true ground area in m2. A post-run filter only (free, instant): it just
        # hides detections client-side, no re-detection. 0 = off / no limit.
        # Merge policy (count distinct objects vs merge tile-split continuous
        # features) is decided automatically per object type and applied silently
        # (no UI control): it only ever changes large >~tile-overlap features, so
        # exposing a toggle would add a knob that does nothing for the common
        # countable objects. See plugin _default_merge_separate / _auto_seam_min_dim.

        # Size filters on ONE compact row (Min ... Max); both hide detections
        # client-side by true ground area (free, instant); 0 = off / no limit.
        _size_row = QHBoxLayout()
        _min_size_lbl = QLabel(tr("Min size:"))
        _min_size_lbl.setStyleSheet("font-size: 11px;")
        self.auto_min_size_spin = QDoubleSpinBox()
        self.auto_min_size_spin.setRange(0.0, 1_000_000.0)
        self.auto_min_size_spin.setDecimals(1)
        self.auto_min_size_spin.setValue(0.0)
        self.auto_min_size_spin.setSuffix(" m²")
        self.auto_min_size_spin.setSpecialValueText(tr("Off"))
        self.auto_min_size_spin.setMinimumWidth(78)
        self.auto_min_size_spin.setMaximumWidth(110)
        self.auto_min_size_spin.setToolTip(tr(
            "Hide detections smaller than this ground area. Use it to drop tiny "
            "noise blobs. 0 = keep all."))
        _max_size_lbl = QLabel(tr("Max size:"))
        _max_size_lbl.setStyleSheet("font-size: 11px;")
        self.auto_max_size_spin = QDoubleSpinBox()
        self.auto_max_size_spin.setRange(0.0, 10_000_000.0)
        self.auto_max_size_spin.setDecimals(1)
        self.auto_max_size_spin.setValue(0.0)
        self.auto_max_size_spin.setSuffix(" m²")
        self.auto_max_size_spin.setSpecialValueText(tr("No limit"))
        self.auto_max_size_spin.setMinimumWidth(78)
        self.auto_max_size_spin.setMaximumWidth(110)
        self.auto_max_size_spin.setToolTip(tr(
            "Hide detections larger than this ground area. 0 = no limit."))
        _size_row.addWidget(_min_size_lbl)
        _size_row.addWidget(self.auto_min_size_spin)
        _size_row.addStretch()
        _size_row.addWidget(_max_size_lbl)
        _size_row.addWidget(self.auto_max_size_spin)

        # Assembly, mirroring the Manual refine panel's sectioned card. The
        # one-click shape switches lead (the preset's headline decisions, and
        # the first thing to glance at), the size filter follows, and the
        # numeric outline fine-tuning closes the section.
        _shape_layout.addWidget(_shape_section_label(tr("Shape")))
        _shape_layout.addLayout(_ortho_row)
        _shape_layout.addLayout(_round_row)
        _shape_layout.addLayout(_fill_row)
        _shape_layout.addWidget(_shape_section_label(tr("Size")))
        _shape_layout.addLayout(_size_row)
        _shape_layout.addWidget(_shape_section_label(tr("Outline")))
        _shape_layout.addLayout(_simplify_hdr)
        _shape_layout.addLayout(_clean_hdr)
        _shape_layout.addLayout(_expand_row)

        self.auto_shape_content.setVisible(self._auto_shape_expanded)
        _review_layout.addWidget(self.auto_shape_content)

        # Debug-only: show the segmentation tile grid over the result. This is a
        # developer aid (inspect which tile a detection came from / spot seam
        # issues), NOT an end-user control, so it is hidden unless the
        # TerraLab/auto_debug_tiles QSettings flag is on. The checkbox still
        # exists (set_auto_review_active resets it), it just isn't shown.
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
        _review_layout.addWidget(self._auto_tiles_debug_row)
        from qgis.PyQt.QtCore import QSettings as _QSettingsDbg
        self._auto_tiles_debug_row.setVisible(
            _QSettingsDbg().value("TerraLab/auto_debug_tiles", False, type=bool))

        # Wire the size filters to emit auto_refine_changed (confidence has its
        # own debounced re-filter path). Shape sliders moved to Manual mode.
        self.auto_min_size_spin.valueChanged.connect(
            lambda _v: self.auto_refine_changed.emit())
        self.auto_max_size_spin.valueChanged.connect(
            lambda _v: self.auto_refine_changed.emit())

        # --- "Keep this result": the commit group. Refine comes BEFORE Export
        # because that is the order the work happens (check -> fix -> export);
        # its outline-blue style ("still editing") keeps it clearly subordinate
        # to the one green element on the panel, the Export primary.
        _review_layout.addWidget(_make_review_group_label(tr("Keep this result")))
        self.auto_refine_in_manual_btn = QPushButton(
            "✎  " + tr("Refine in Manual mode"))
        self.auto_refine_in_manual_btn.setStyleSheet(_BTN_OUTLINE_BLUE)
        self.auto_refine_in_manual_btn.setMinimumHeight(36)
        self.auto_refine_in_manual_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_refine_in_manual_btn.setToolTip(tr(
            "Open these detections in Manual mode to fix specific objects "
            "point-by-point, then come back and export."))
        self.auto_refine_in_manual_btn.clicked.connect(
            self.auto_refine_in_manual_requested.emit)
        _review_layout.addWidget(self.auto_refine_in_manual_btn)

        # Export: the primary, final commit. Names the outcome (a saved layer)
        # instead of the vague "Finish"; taller than every other control so the
        # finish line is unmistakable.
        self.auto_export_btn = QPushButton(tr("Export {n} polygons").format(n=0))
        self.auto_export_btn.setStyleSheet(_BTN_GREEN)
        self.auto_export_btn.setMinimumHeight(44)
        self.auto_export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_export_btn.clicked.connect(self.auto_export_requested.emit)
        _review_layout.addWidget(self.auto_export_btn)

        # Exemplar nudge: shown only when the run's scores were bottom-heavy
        # AND no example was drawn. Clicking it routes through Adjust and run
        # again and arms the example draw, so it sits with the "Start over"
        # group it triggers. Hidden by default; the plugin drives it via
        # show/hide_auto_exemplar_nudge. Styled as a tinted callout, not a
        # muted text link: the link form shipped at launch and no user ever
        # clicked it, so the one lever that most improves weak runs (an
        # example cuts empty tiles by two thirds) needs to read as a button.
        self.auto_exemplar_nudge_link = QPushButton("")
        self.auto_exemplar_nudge_link.setStyleSheet(
            "QPushButton { background: rgba(30,136,229,0.10);"
            " border: 1px solid rgba(30,136,229,0.35); border-radius: 6px;"
            " color: palette(text); font-size: 12px; text-align: left;"
            " padding: 8px 10px; }"
            "QPushButton:hover { background: rgba(30,136,229,0.20); }")
        self.auto_exemplar_nudge_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_exemplar_nudge_link.setVisible(False)
        self.auto_exemplar_nudge_link.clicked.connect(
            self.auto_exemplar_retry_requested.emit)

        # --- "Start over": the two escape hatches, deliberately quiet so the
        # eye lands on the commit group above. Retry keeps the zone, references
        # and settings (non-destructive re-run, new credits); Exit discards.
        _review_layout.addWidget(_make_review_group_label(tr("Start over")))
        _review_layout.addWidget(self.auto_exemplar_nudge_link)
        self.auto_retry_btn = QPushButton("↻  " + tr("Adjust and run again"))
        self.auto_retry_btn.setStyleSheet(_BTN_GHOST)
        self.auto_retry_btn.setMinimumHeight(32)
        self.auto_retry_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_retry_btn.setToolTip(tr(
            "Go back to your zone, references and settings to adjust and detect "
            "again. Nothing is saved."))
        self.auto_retry_btn.clicked.connect(self.auto_retry_requested.emit)
        self.auto_review_exit_btn = QPushButton(tr("Exit"))
        self.auto_review_exit_btn.setStyleSheet(_BTN_GHOST)
        self.auto_review_exit_btn.setMinimumHeight(32)
        self.auto_review_exit_btn.setMaximumWidth(72)
        self.auto_review_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_review_exit_btn.clicked.connect(self.auto_review_exit_requested.emit)
        _review_actions_row = QHBoxLayout()
        _review_actions_row.setContentsMargins(0, 0, 0, 0)
        _review_actions_row.addWidget(self.auto_retry_btn, 1)
        _review_actions_row.addWidget(self.auto_review_exit_btn, 0)
        _review_layout.addLayout(_review_actions_row)

        parent_layout.addWidget(self.auto_review_panel)

    # -- Collapsible "Refine detections" section state ---------------------

    def _refresh_auto_shape_header(self) -> None:
        """Chevron + title for the collapsible section header (text swap only,
        no layout jump)."""
        arrow = "▾" if self._auto_shape_expanded else "▸"
        self.auto_shape_toggle_btn.setText(
            arrow + " " + tr("Refine detections").upper())

    def _on_auto_shape_toggle(self) -> None:
        """Header clicked: flip the section. Pure setVisible, so collapsing or
        expanding never emits a control signal or steals focus."""
        self.set_auto_shape_expanded(not self._auto_shape_expanded)

    def set_auto_shape_expanded(self, expanded: bool) -> None:
        """Programmatically expand/collapse the Refine detections section
        (set_auto_review_active collapses it for every NEW review)."""
        self._auto_shape_expanded = bool(expanded)
        try:
            self.auto_shape_content.setVisible(self._auto_shape_expanded)
            self._refresh_auto_shape_header()
        except (RuntimeError, AttributeError):
            pass

    # The empty-run guidance box (title + credits + four fixes + zero-assist
    # chips + a dismissible guide nudge) is gone: a finished run
    # that finds nothing now shows a single quiet status line at the top of
    # step 2 ("No detection in this zone.") and leaves the user on the normal
    # prompt screen to adjust and run again or Exit. See _on_auto_zero_detections.
