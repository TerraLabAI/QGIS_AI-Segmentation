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
    REFINE_CLEAN_DEFAULT,
    REFINE_EXPAND_DEFAULT,
    REFINE_FILL_HOLES_DEFAULT,
    REFINE_FILL_HOLES_MAX_M2_DEFAULT,
    REFINE_MAX_SIZE_M2_DEFAULT,
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_POINTS_PCT_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
    REFINE_SMOOTH_ITERATIONS,
)
from .styles import (
    _CARD_QSS,
    _SECTION_TOGGLE_QSS,
    _settings_zone,
)


def _refine_row_label(text: str, tooltip: str) -> QLabel:
    """Left label of one refine row: the same quiet 11px as the Automatic
    review's rows, so the two panels line up."""
    lbl = QLabel(text)
    lbl.setStyleSheet("font-size: 11px;")
    lbl.setToolTip(tooltip)
    return lbl


def _refine_control_row(label: QLabel, control) -> QHBoxLayout:
    """One row: label on the left, control on the right-aligned rail."""
    row = QHBoxLayout()
    row.addWidget(label)
    row.addStretch()
    row.addWidget(control)
    return row


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

        The controls sit in the same three sub-cards as the Automatic review's
        Shapes step (Shape / Outline / Size) and carry its wording, so one
        setting reads the same in both modes.
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

        _refine_qss = _CARD_QSS.format(name="refineContentWidget")
        _refine_qss += "QLabel { background: transparent; border: none; }"
        _refine_qss += _cb_qss
        self.refine_content_widget.setStyleSheet(_refine_qss)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        # 9px between sibling sub-cards; each sub-card keeps 6 inside.
        refine_content_layout.setSpacing(9)

        # ── Shape: right angles, round corners, fill holes ──

        # Right angles: orthogonalize for man-made shapes. First, as in the
        # Automatic review, because it decides what the rows under it may do.
        ortho_label = _refine_row_label(tr("Right angles"), tr(
            "Snap walls to right angles, 45 degree walls included. Made for "
            "buildings, pools and solar panels. A shape it would distort is "
            "left as it is."))
        self.right_angles_checkbox = QCheckBox()
        self.right_angles_checkbox.setToolTip(ortho_label.toolTip())
        self.right_angles_checkbox.setChecked(REFINE_ORTHO_DEFAULT)
        ortho_row = _refine_control_row(ortho_label, self.right_angles_checkbox)

        round_label = _refine_row_label(tr("Round corners"), tr(
            "Round corners for natural shapes like trees and bushes. "
            "Lower Points for smoother results."))
        self.round_corners_checkbox = QCheckBox()
        self.round_corners_checkbox.setToolTip(round_label.toolTip())
        self.round_corners_checkbox.setChecked(REFINE_SMOOTH_DEFAULT > 0)
        round_row = _refine_control_row(
            round_label, self.round_corners_checkbox)

        fill_label = _refine_row_label(
            tr("Fill holes"), tr("Fill interior holes in the selection"))
        self.fill_holes_checkbox = QCheckBox()
        self.fill_holes_checkbox.setChecked(REFINE_FILL_HOLES_DEFAULT)
        self.fill_holes_checkbox.setToolTip(fill_label.toolTip())
        fill_row = _refine_control_row(fill_label, self.fill_holes_checkbox)

        # Up to (true ground m2): the size threshold ArcGIS "Eliminate Polygon
        # Part" and QGIS native:deleteholes both use. Segment a road and the
        # cars parked on it come back as interior holes, while the median is a
        # real hole; one number tells them apart. The row only shows while Fill
        # holes is on, so an off checkbox stays one line.
        self.fill_holes_max_row = QWidget()
        fill_max_layout = QHBoxLayout(self.fill_holes_max_row)
        fill_max_layout.setContentsMargins(16, 2, 0, 0)
        fill_max_layout.setSpacing(6)
        fill_max_label = _refine_row_label(tr("Up to"), tr(
            "Fill only holes smaller than this ground area. Bigger holes (a "
            "road median, a courtyard) stay open. No limit = fill every hole."))
        fill_max_label.setStyleSheet(
            "font-size: 11px; color: rgba(128, 128, 128, 0.95);")
        self.fill_holes_max_spinbox = QDoubleSpinBox()
        self.fill_holes_max_spinbox.setRange(0.0, 1_000_000.0)
        self.fill_holes_max_spinbox.setDecimals(1)
        self.fill_holes_max_spinbox.setValue(REFINE_FILL_HOLES_MAX_M2_DEFAULT)
        self.fill_holes_max_spinbox.setSuffix(" m²")
        self.fill_holes_max_spinbox.setSpecialValueText(tr("No limit"))
        self.fill_holes_max_spinbox.setToolTip(fill_max_label.toolTip())
        self.fill_holes_max_spinbox.setMinimumWidth(78)
        self.fill_holes_max_spinbox.setMaximumWidth(110)
        fill_max_layout.addWidget(fill_max_label)
        fill_max_layout.addStretch()
        fill_max_layout.addWidget(self.fill_holes_max_spinbox)
        self.fill_holes_max_row.setVisible(REFINE_FILL_HOLES_DEFAULT)

        # ── Outline: points, simplify, trim spikes, grow / shrink ──

        # Points: the share of its own points the outline keeps. A mask traced
        # pixel by pixel carries one point every few centimetres, which is the
        # raster grid showing through, not accuracy. A count, not a distance,
        # which is why it sits above Simplify.
        # Manual has no prompt, so no object class: 100% is the one generic
        # density, where the Automatic review's 100% is the class default.
        points_label = _refine_row_label(tr("Points"), tr(
            "How many of the outline's points to keep, as a share. 100% keeps "
            "the standard density; lower thins further, dropping the smallest "
            "detail first and keeping the corners and the surface. With Right "
            "angles on it runs first, so lowering it gives the squaring "
            "straight walls to work from."))
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setSingleStep(5)
        self.points_spinbox.setRange(1, 100)
        self.points_spinbox.setValue(REFINE_POINTS_PCT_DEFAULT)
        self.points_spinbox.setSuffix(" %")
        self.points_spinbox.setMinimumWidth(62)
        self.points_spinbox.setMaximumWidth(78)
        self.points_spinbox.setToolTip(points_label.toolTip())
        points_row = _refine_control_row(points_label, self.points_spinbox)

        # Simplify (px): the Douglas-Peucker tolerance. A distance, so pushed
        # hard it sweeps whole walls into one slanted chord; it de-noises, and
        # Points is what thins.
        simplify_label = _refine_row_label(tr("Simplify"), tr(
            "Drop points closer than this distance to a straight edge (0 = "
            "off). A distance, not a count: pushed high it can flatten curved "
            "walls. Points is usually the better dial for thinning an "
            "outline."))
        self.simplify_spinbox = QDoubleSpinBox()
        self.simplify_spinbox.setRange(0.0, 1000.0)
        self.simplify_spinbox.setDecimals(1)
        self.simplify_spinbox.setSingleStep(0.5)
        self.simplify_spinbox.setValue(REFINE_SIMPLIFY_DEFAULT)
        self.simplify_spinbox.setSuffix(" px")
        self.simplify_spinbox.setSpecialValueText(tr("Off"))
        self.simplify_spinbox.setMinimumWidth(62)
        self.simplify_spinbox.setMaximumWidth(78)
        self.simplify_spinbox.setToolTip(simplify_label.toolTip())
        simplify_row = _refine_control_row(
            simplify_label, self.simplify_spinbox)

        # Trim spikes (morphological opening, px; 0 = off): strips thin attached
        # fringe. Unlike Minimum size, which drops SEPARATE small parts, this
        # removes noise that belongs to the same polygon.
        clean_label = _refine_row_label(tr("Trim spikes"), tr(
            "Shave thin spikes and ragged bits off each shape's outline. It "
            "leaves the main body alone; higher values trim more. 0 = off."))
        self.clean_edges_spinbox = QDoubleSpinBox()
        self.clean_edges_spinbox.setRange(0.0, 100.0)
        self.clean_edges_spinbox.setDecimals(1)
        self.clean_edges_spinbox.setSingleStep(0.5)
        self.clean_edges_spinbox.setValue(REFINE_CLEAN_DEFAULT)
        self.clean_edges_spinbox.setSuffix(" px")
        self.clean_edges_spinbox.setSpecialValueText(tr("Off"))
        self.clean_edges_spinbox.setToolTip(clean_label.toolTip())
        self.clean_edges_spinbox.setMinimumWidth(62)
        self.clean_edges_spinbox.setMaximumWidth(78)
        clean_row = _refine_control_row(clean_label, self.clean_edges_spinbox)

        expand_label = _refine_row_label(tr("Grow / shrink"), tr(
            "Positive = grow outward, negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(REFINE_EXPAND_DEFAULT)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setToolTip(expand_label.toolTip())
        self.expand_spinbox.setMinimumWidth(62)
        self.expand_spinbox.setMaximumWidth(78)
        expand_row = _refine_control_row(expand_label, self.expand_spinbox)

        # Right angles is a guided geometry mode: extra generic cleanup can
        # erase narrow parts and corner rounding reverses what was asked for.
        # Points and Simplify are NOT blocked (the squaring runs on the outline
        # the point budget produced), matching the Automatic review.
        self._refine_right_angle_conflicts = (
            clean_label,
            self.clean_edges_spinbox,
            round_label,
            self.round_corners_checkbox,
        )
        self._refine_right_angle_conflict_tooltips = tuple(
            (widget, widget.toolTip())
            for widget in self._refine_right_angle_conflicts)

        # ── Size: hide parts outside the window (true ground m2, 0 = off) ──
        # The same filters as the Automatic review, whose values seed these
        # during a Refine-in-Manual handoff. (The old pixel min-area is
        # separate: auto-computed per crop, see plugin._compute_auto_min_area.)
        size_row = QHBoxLayout()
        size_row.setContentsMargins(0, 0, 0, 0)
        size_row.setSpacing(10)
        # "Parts", not "detections": Manual filters the pieces of the ONE
        # selection under the cursor, where the Automatic review filters whole
        # detected objects.
        min_size_label = _refine_row_label(tr("Minimum"), tr(
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
        max_size_label = _refine_row_label(tr("Maximum"), tr(
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
        size_row.addWidget(min_size_label)
        size_row.addWidget(self.min_size_spinbox)
        size_row.addStretch()
        size_row.addWidget(max_size_label)
        size_row.addWidget(self.max_size_spinbox)

        # Assembly: three sub-cards, the same cut and the same order as the
        # Automatic review's Shapes step, so the panel reads as separated
        # groups instead of one flat list of rows.
        refine_content_layout.addWidget(_settings_zone(
            "refineZoneShape", tr("Shape"),
            tr("how the outline is styled"), [
                ortho_row,
                round_row,
                fill_row,
                self.fill_holes_max_row,
            ]))
        refine_content_layout.addWidget(_settings_zone(
            "refineZoneOutline", tr("Outline"),
            tr("fine-tune the edges"), [
                points_row,
                simplify_row,
                clean_row,
                expand_row,
            ]))
        refine_content_layout.addWidget(_settings_zone(
            "refineZoneSize", tr("Size"),
            tr("hide anything outside this range"), [
                size_row,
            ]))

        refine_layout.addWidget(self.refine_content_widget)
        self.refine_content_widget.setVisible(self._refine_expanded)
        self._refresh_refine_header()

        # Connect signals
        self.points_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.clean_edges_spinbox.valueChanged.connect(self._on_refine_changed)
        self.round_corners_checkbox.stateChanged.connect(self._on_refine_changed)
        self.right_angles_checkbox.stateChanged.connect(self._on_refine_changed)
        self.right_angles_checkbox.stateChanged.connect(
            self._sync_refine_right_angle_controls)
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(
            self._sync_fill_holes_max_row)
        self.fill_holes_max_spinbox.valueChanged.connect(self._on_refine_changed)
        self.min_size_spinbox.valueChanged.connect(self._on_refine_changed)
        self.max_size_spinbox.valueChanged.connect(self._on_refine_changed)

        self._sync_refine_right_angle_controls()
        parent_layout.addWidget(self.refine_group)

    def _sync_fill_holes_max_row(self, _state=None) -> None:
        """Show the size threshold only while Fill holes is on (an irrelevant
        control is hidden, never greyed)."""
        row = getattr(self, "fill_holes_max_row", None)
        if row is not None:
            row.setVisible(self.fill_holes_checkbox.isChecked())

    def _sync_refine_right_angle_controls(self, _state=None) -> None:
        """Make incompatible Shape controls unavailable with Right angles.

        The Automatic review's rule (_sync_auto_right_angle_controls), applied
        to the Manual panel. _emit_refine_changed enforces the same thing on the
        values, so a disabled widget can never leave an old setting active.
        """
        ortho = getattr(self, "right_angles_checkbox", None)
        enabled = not bool(ortho is not None and ortho.isChecked())
        blocked_tip = tr(
            "Unavailable while Right angles is on. Turn it off to adjust this "
            "setting.")
        for widget, normal_tip in getattr(
                self, "_refine_right_angle_conflict_tooltips", ()):
            try:
                widget.setEnabled(enabled)
                widget.setToolTip(normal_tip if enabled else blocked_tip)
            except (RuntimeError, AttributeError):
                pass
        # Curving an outline after it has been squared is contradictory. Clear
        # the state as well as disabling the control, so toggling Right angles
        # never leaves a hidden rounding pass in the preview.
        round_corners = getattr(self, "round_corners_checkbox", None)
        if not enabled and round_corners is not None:
            try:
                round_corners.blockSignals(True)
                round_corners.setChecked(False)
            except (RuntimeError, AttributeError):
                pass
            finally:
                try:
                    round_corners.blockSignals(False)
                except (RuntimeError, AttributeError):
                    pass

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

        The size filter, the fill-holes threshold and the outline budget go out
        FIRST (their handlers only store the values), so the
        refine_settings_changed handler that follows repaints once with
        everything fresh.
        """
        right_angles = self.right_angles_checkbox.isChecked()
        self.size_filter_changed.emit(
            float(self.min_size_spinbox.value()),
            float(self.max_size_spinbox.value()),
        )
        self.fill_holes_size_changed.emit(
            float(self.fill_holes_max_spinbox.value()))
        # Right angles blocks Trim spikes and Round corners in the panel; force
        # their values off here too, so a value set before the tick can never
        # stay active behind a disabled control.
        self.clean_edges_changed.emit(
            0.0 if right_angles else float(self.clean_edges_spinbox.value()))
        self.outline_budget_changed.emit(
            float(self.simplify_spinbox.value()),
            int(self.points_spinbox.value()),
        )
        # The first slot is legacy and no longer read: the float tolerance
        # travels on outline_budget_changed.
        self.refine_settings_changed.emit(
            int(round(self.simplify_spinbox.value())),
            0 if right_angles else (
                REFINE_SMOOTH_ITERATIONS
                if self.round_corners_checkbox.isChecked() else 0),
            self.expand_spinbox.value(),
            self.fill_holes_checkbox.isChecked(),
            right_angles,
        )

    def get_refine_points_pct(self) -> int:
        """Points dial: the share of its own points an outline keeps (1-100)."""
        try:
            return int(self.points_spinbox.value())
        except (RuntimeError, AttributeError):
            return REFINE_POINTS_PCT_DEFAULT

    def get_refine_simplify_px(self) -> float:
        """Simplify tolerance in crop pixels (0 = off)."""
        try:
            return float(self.simplify_spinbox.value())
        except (RuntimeError, AttributeError):
            return float(REFINE_SIMPLIFY_DEFAULT)

    def reset_refine_sliders(self):
        """Reset refinement controls to default values without emitting signals."""
        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox,
                  self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(True)

        self.points_spinbox.setValue(REFINE_POINTS_PCT_DEFAULT)
        self.simplify_spinbox.setValue(REFINE_SIMPLIFY_DEFAULT)
        self.clean_edges_spinbox.setValue(REFINE_CLEAN_DEFAULT)
        self.round_corners_checkbox.setChecked(REFINE_SMOOTH_DEFAULT > 0)
        self.right_angles_checkbox.setChecked(REFINE_ORTHO_DEFAULT)
        self.expand_spinbox.setValue(REFINE_EXPAND_DEFAULT)
        self.fill_holes_checkbox.setChecked(REFINE_FILL_HOLES_DEFAULT)
        self.fill_holes_max_spinbox.setValue(REFINE_FILL_HOLES_MAX_M2_DEFAULT)
        self.min_size_spinbox.setValue(REFINE_MIN_SIZE_M2_DEFAULT)
        self.max_size_spinbox.setValue(REFINE_MAX_SIZE_M2_DEFAULT)

        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox,
                  self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(False)
        self._sync_fill_holes_max_row()
        self._sync_refine_right_angle_controls()

    def set_refine_values(self, simplify: float, smooth: int, expand: int,
                          fill_holes: bool, min_area: int | None = None,
                          right_angles: bool = False,
                          fill_holes_max_m2: float | None = None,
                          clean: float | None = None,
                          points_pct: int | None = None):
        """Set refine slider values without emitting signals.

        min_area is kept in the signature for backward compatibility with
        stored polygon metadata but no longer touches the UI.
        simplify is the tolerance in crop pixels (0 = off).
        fill_holes_max_m2 is the fill-holes size threshold (ground m2, 0 = every
        hole); None leaves the current value, so old callers stay correct.
        clean is the Trim-spikes opening distance (px, 0 = off); None leaves
        the current value. points_pct is the Points dial (1-100); None leaves
        the current value.
        """
        del min_area  # unused since the spinbox was removed
        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(float(simplify))
        self.round_corners_checkbox.setChecked(smooth > 0)
        self.right_angles_checkbox.setChecked(bool(right_angles))
        self.expand_spinbox.setValue(expand)
        self.fill_holes_checkbox.setChecked(fill_holes)
        if fill_holes_max_m2 is not None:
            self.fill_holes_max_spinbox.setValue(
                max(0.0, float(fill_holes_max_m2)))
        if clean is not None:
            self.clean_edges_spinbox.setValue(max(0.0, float(clean)))
        if points_pct is not None:
            self.points_spinbox.setValue(
                max(1, min(100, int(points_pct))))

        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox):
            w.blockSignals(False)
        self._sync_fill_holes_max_row()
        self._sync_refine_right_angle_controls()

    def set_fill_holes_max_value(self, max_m2: float) -> None:
        """Set the fill-holes size threshold without emitting signals
        (ground m2, 0 = fill every hole)."""
        self.fill_holes_max_spinbox.blockSignals(True)
        self.fill_holes_max_spinbox.setValue(max(0.0, float(max_m2 or 0.0)))
        self.fill_holes_max_spinbox.blockSignals(False)

    def set_size_filter_values(self, min_m2: float, max_m2: float) -> None:
        """Set the Min/Max size filters without emitting signals (0 = off)."""
        for w in (self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(True)
        self.min_size_spinbox.setValue(max(0.0, float(min_m2 or 0.0)))
        self.max_size_spinbox.setValue(max(0.0, float(max_m2 or 0.0)))
        for w in (self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(False)
