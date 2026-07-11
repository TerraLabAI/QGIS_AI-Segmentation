"""Refine panel: build, expand/collapse, slider values and resets.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations


from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


from ...core.i18n import tr
from ...core.review_defaults import (
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_MAX_SIZE_M2_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
)
from .styles import (
    _CARD_QSS,
    _SECTION_TOGGLE_QSS,
    _micro_header,
)


class DockRefineMixin:
    """Refine panel: build, expand/collapse, slider values and resets."""

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel.

        The panel is a framed header-BUTTON (chevron + title) over a card of
        controls, the design-system collapsible pattern shared with the
        Automatic review. The title is contextual: base Manual calls it "Refine
        selection"; a Refine-in-Manual handoff retitles it "Shape settings"
        (per-polygon controls). ``refine_group`` stays the container name that
        state.py shows/hides.
        """
        self._refine_panel_title = tr("Refine selection")
        self.refine_group = QWidget()
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(6)
        refine_layout.setContentsMargins(0, 0, 0, 0)

        # Framed header-button with a chevron: a full-width control that reads
        # as clickable at a glance, replacing the old click-position hit-test.
        self.refine_toggle_btn = QPushButton()
        self.refine_toggle_btn.setStyleSheet(_SECTION_TOGGLE_QSS)
        self.refine_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Never steal focus from the spinboxes/checkboxes on toggle.
        self.refine_toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refine_toggle_btn.clicked.connect(self._on_refine_toggle_clicked)
        refine_layout.addWidget(self.refine_toggle_btn)

        # Content card (standard _CARD_QSS look) shown/hidden by the header.
        self.refine_content_widget = QWidget()
        self.refine_content_widget.setObjectName("refineContentWidget")
        self.refine_content_widget.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)

        # Shared visible checkbox indicators (off = outlined box, on = blue
        # check): the old inline painter drew the unchecked state as a plain
        # background square, i.e. invisible.
        from .widgets import checkbox_indicator_qss
        _cb_qss = checkbox_indicator_qss(self)

        self.refine_content_widget.setStyleSheet(
            _CARD_QSS.format(name="refineContentWidget")
            + "QLabel { background: transparent; border: none; }"
            + _cb_qss)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        refine_content_layout.setSpacing(6)

        # ── Outline section ──
        refine_content_layout.addWidget(_micro_header(tr("Outline")))

        # 1. Simplify outline: SpinBox (0 to 1000) - reduces small variations
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(tr("Reduce small variations in the outline (0 = no change)"))
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 1000)
        self.simplify_spinbox.setValue(REFINE_SIMPLIFY_DEFAULT)
        self.simplify_spinbox.setMinimumWidth(55)
        self.simplify_spinbox.setMaximumWidth(70)
        simplify_layout.addWidget(simplify_label)
        simplify_layout.addStretch()
        simplify_layout.addWidget(self.simplify_spinbox)
        refine_content_layout.addLayout(simplify_layout)

        # 2. Round corners: Label + Checkbox aligned right (like spinbox rows)
        round_layout = QHBoxLayout()
        round_label = QLabel(tr("Round corners:"))
        round_label.setToolTip(
            tr("Round corners for natural shapes like trees and bushes. "
               "Increase 'Simplify outline' for smoother results."))
        self.round_corners_checkbox = QCheckBox()
        self.round_corners_checkbox.setToolTip(round_label.toolTip())
        self.round_corners_checkbox.setChecked(REFINE_SMOOTH_DEFAULT > 0)
        round_layout.addWidget(round_label)
        round_layout.addStretch()
        round_layout.addWidget(self.round_corners_checkbox)
        refine_content_layout.addLayout(round_layout)

        # 2b. Right angles: orthogonalize for man-made shapes (mirrors the
        # Automatic review's control, and inherits its state from the review
        # during a Refine-in-Manual handoff).
        ortho_layout = QHBoxLayout()
        ortho_label = QLabel(tr("Right angles:"))
        ortho_label.setToolTip(tr(
            "Snap edges to 90 degrees for man-made shapes like buildings, "
            "pools and solar panels."))
        self.right_angles_checkbox = QCheckBox()
        self.right_angles_checkbox.setToolTip(ortho_label.toolTip())
        self.right_angles_checkbox.setChecked(REFINE_ORTHO_DEFAULT)
        ortho_layout.addWidget(ortho_label)
        ortho_layout.addStretch()
        ortho_layout.addWidget(self.right_angles_checkbox)
        refine_content_layout.addLayout(ortho_layout)

        # ── Selection section ──
        refine_content_layout.addWidget(_micro_header(tr("Selection")))

        # 3. Expand/Contract: SpinBox with +/- buttons (-1000 to +1000)
        expand_layout = QHBoxLayout()
        expand_label = QLabel(tr("Expand/Contract:"))
        expand_label.setToolTip(tr("Positive = expand outward, Negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(REFINE_EXPAND_DEFAULT)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setMinimumWidth(55)
        self.expand_spinbox.setMaximumWidth(70)
        expand_layout.addWidget(expand_label)
        expand_layout.addStretch()
        expand_layout.addWidget(self.expand_spinbox)
        refine_content_layout.addLayout(expand_layout)

        # 4. Fill holes: Label + Checkbox aligned right (like spinbox rows)
        fill_layout = QHBoxLayout()
        fill_label = QLabel(tr("Fill holes:"))
        fill_label.setToolTip(tr("Fill interior holes in the selection"))
        self.fill_holes_checkbox = QCheckBox()
        self.fill_holes_checkbox.setChecked(REFINE_FILL_HOLES_DEFAULT)
        self.fill_holes_checkbox.setToolTip(fill_label.toolTip())
        fill_layout.addWidget(fill_label)
        fill_layout.addStretch()
        fill_layout.addWidget(self.fill_holes_checkbox)
        refine_content_layout.addLayout(fill_layout)

        # 5./6. Min/Max size (true ground m2, 0 = off): drop parts of the
        # selection outside the window. The same size filters as the Automatic
        # review, whose values seed these during a Refine-in-Manual handoff.
        # (The old pixel min-area is separate: auto-computed per crop, see
        # plugin._compute_auto_min_area.)
        min_size_layout = QHBoxLayout()
        min_size_label = QLabel(tr("Min size:"))
        min_size_label.setToolTip(tr(
            "Hide parts smaller than this ground area. Use it to drop tiny "
            "noise blobs. 0 = keep all."))
        self.min_size_spinbox = QDoubleSpinBox()
        self.min_size_spinbox.setRange(0.0, 1_000_000.0)
        self.min_size_spinbox.setDecimals(1)
        self.min_size_spinbox.setValue(REFINE_MIN_SIZE_M2_DEFAULT)
        self.min_size_spinbox.setSuffix(" m²")
        self.min_size_spinbox.setSpecialValueText(tr("Off"))
        self.min_size_spinbox.setToolTip(min_size_label.toolTip())
        self.min_size_spinbox.setMinimumWidth(78)
        self.min_size_spinbox.setMaximumWidth(110)
        min_size_layout.addWidget(min_size_label)
        min_size_layout.addStretch()
        min_size_layout.addWidget(self.min_size_spinbox)
        refine_content_layout.addLayout(min_size_layout)

        max_size_layout = QHBoxLayout()
        max_size_label = QLabel(tr("Max size:"))
        max_size_label.setToolTip(tr(
            "Hide parts larger than this ground area. 0 = no limit."))
        self.max_size_spinbox = QDoubleSpinBox()
        self.max_size_spinbox.setRange(0.0, 10_000_000.0)
        self.max_size_spinbox.setDecimals(1)
        self.max_size_spinbox.setValue(REFINE_MAX_SIZE_M2_DEFAULT)
        self.max_size_spinbox.setSuffix(" m²")
        self.max_size_spinbox.setSpecialValueText(tr("No limit"))
        self.max_size_spinbox.setToolTip(max_size_label.toolTip())
        self.max_size_spinbox.setMinimumWidth(78)
        self.max_size_spinbox.setMaximumWidth(110)
        max_size_layout.addWidget(max_size_label)
        max_size_layout.addStretch()
        max_size_layout.addWidget(self.max_size_spinbox)
        refine_content_layout.addLayout(max_size_layout)

        refine_layout.addWidget(self.refine_content_widget)
        self.refine_content_widget.setVisible(self._refine_expanded)
        self._refresh_refine_header()

        # Connect signals
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.round_corners_checkbox.stateChanged.connect(self._on_refine_changed)
        self.right_angles_checkbox.stateChanged.connect(self._on_refine_changed)
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_changed)
        self.min_size_spinbox.valueChanged.connect(self._on_refine_changed)
        self.max_size_spinbox.valueChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _refresh_refine_header(self) -> None:
        """Chevron + normal-case title on the collapsible header (text swap
        only)."""
        arrow = "▾" if self._refine_expanded else "▸"
        self.refine_toggle_btn.setText(
            arrow + " " + self._refine_panel_title)

    def _on_refine_toggle_clicked(self) -> None:
        """Header clicked: flip the panel. Pure setVisible, so it never emits a
        control signal or steals focus."""
        self._refine_expanded = not self._refine_expanded
        self._apply_refine_toggle(self._refine_expanded)

    def set_refine_panel_title(self, title: str) -> None:
        """Retitle the panel (keeps the current collapse chevron)."""
        self._refine_panel_title = title
        self._refresh_refine_header()

    def set_refine_collapsed(self, collapsed: bool) -> None:
        """Force the panel collapsed/expanded (immediate)."""
        self._refine_expanded = not collapsed
        self._apply_refine_toggle(self._refine_expanded)

    def _apply_refine_toggle(self, expanded):
        """Show/hide the content card and sync the header chevron."""
        self.refine_content_widget.setVisible(expanded)
        self._refresh_refine_header()

    def _on_refine_changed(self, value=None):
        """Handle refine control changes with debounce."""
        self._refine_debounce_timer.start(150)

    def _emit_refine_changed(self):
        """Emit the refine settings changed signals after debounce.

        The size filter goes out FIRST (its handler only stores the values),
        so the refine_settings_changed handler that follows repaints once with
        everything fresh."""
        self.size_filter_changed.emit(
            float(self.min_size_spinbox.value()),
            float(self.max_size_spinbox.value()),
        )
        self.refine_settings_changed.emit(
            self.simplify_spinbox.value(),
            5 if self.round_corners_checkbox.isChecked() else 0,
            self.expand_spinbox.value(),
            self.fill_holes_checkbox.isChecked(),
            self.right_angles_checkbox.isChecked(),
        )

    def reset_refine_sliders(self):
        """Reset refinement controls to default values without emitting signals."""
        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(REFINE_SIMPLIFY_DEFAULT)
        self.round_corners_checkbox.setChecked(REFINE_SMOOTH_DEFAULT > 0)
        self.right_angles_checkbox.setChecked(REFINE_ORTHO_DEFAULT)
        self.expand_spinbox.setValue(REFINE_EXPAND_DEFAULT)
        self.fill_holes_checkbox.setChecked(REFINE_FILL_HOLES_DEFAULT)
        self.min_size_spinbox.setValue(REFINE_MIN_SIZE_M2_DEFAULT)
        self.max_size_spinbox.setValue(REFINE_MAX_SIZE_M2_DEFAULT)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(False)

    def set_refine_values(self, simplify: int, smooth: int, expand: int,
                          fill_holes: bool, min_area: int | None = None,
                          right_angles: bool = False):
        """Set refine slider values without emitting signals.

        min_area is kept in the signature for backward compatibility with
        stored polygon metadata but no longer touches the UI.
        """
        del min_area  # unused since the spinbox was removed
        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(simplify)
        self.round_corners_checkbox.setChecked(smooth > 0)
        self.right_angles_checkbox.setChecked(bool(right_angles))
        self.expand_spinbox.setValue(expand)
        self.fill_holes_checkbox.setChecked(fill_holes)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(False)

    def set_size_filter_values(self, min_m2: float, max_m2: float) -> None:
        """Set the Min/Max size filters without emitting signals (0 = off)."""
        for w in (self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(True)
        self.min_size_spinbox.setValue(max(0.0, float(min_m2 or 0.0)))
        self.max_size_spinbox.setValue(max(0.0, float(max_m2 or 0.0)))
        for w in (self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(False)
