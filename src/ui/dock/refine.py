"""Refine panel: build, expand/collapse, slider values and resets.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations

from contextlib import suppress

from qgis.PyQt.QtCore import QSettings, Qt
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
from ...core.qt_compat import safe_single_shot
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
from .font_scale import scale_px_length
from .refine_persistence import (
    apply_refine_start_values,
    capture_refine_settings,
    refine_setting_name_for,
    refine_start_values,
    remember_refine_settings,
)
from .styles import (
    _CARD_JOINED_QSS,
    _SECTION_TOGGLE_OPEN_QSS,
    _SECTION_TOGGLE_QSS,
    _settings_zone,
)

#: How long a refine NUMBER must sit still before the outline is reshaped, in
#: milliseconds. It exists to absorb a held-down or typed-into spinbox, not to
#: pace the redraw: the shape itself is memoized (ui/plugin/manual_shape_cache),
#: so the settled change costs a few milliseconds. Bounds keep a served value
#: usable: below the floor a held arrow key reshapes on every step, above the
#: ceiling the outline visibly trails the number.
REFINE_SETTLE_DEFAULT_MS = 90
_MIN_REFINE_SETTLE_MS = 30
_MAX_REFINE_SETTLE_MS = 400


def refine_settle_ms() -> int:
    """The settle time in force. Server-tunable, cache-only, never raises."""
    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range(
            "ui.refine_settle_ms", REFINE_SETTLE_DEFAULT_MS,
            _MIN_REFINE_SETTLE_MS, _MAX_REFINE_SETTLE_MS))
    except Exception:  # noqa: BLE001 -- a settle time is best-effort  # nosec B110
        return REFINE_SETTLE_DEFAULT_MS


#: QSettings key for the More settings section's open/closed state. Same
#: "AISegmentation/" group as the dismissed hints. Never rename the literal:
#: it sits in the user's QGIS profile.
_REFINE_MORE_EXPANDED_KEY = "AISegmentation/refine/more_expanded"


def _read_refine_more_expanded() -> bool:
    """Whether the long tail opens closed or open. Closed for a new user."""
    try:
        return bool(QSettings().value(
            _REFINE_MORE_EXPANDED_KEY, False, type=bool))
    except Exception:  # noqa: BLE001 -- an unreadable setting is the default
        return False


def _write_refine_more_expanded(expanded: bool) -> None:
    """Remember the long tail's state for the next session."""
    try:
        QSettings().setValue(_REFINE_MORE_EXPANDED_KEY, bool(expanded))
    except Exception:  # noqa: BLE001 -- a preference is best-effort  # nosec B110
        pass


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
        Automatic review. Open, the two share one border and read as a single
        box: the card is what the header opens, not a neighbour of it.

        The title is contextual: base Manual calls it "Outline settings";
        a Refine-in-Manual handoff retitles it "Shape settings" (per-polygon
        controls). ``refine_group`` stays the container name that state.py
        shows/hides.

        Inside: the two toggles users actually touch (Right angles, Round
        corners) at top level, then a collapsed More settings section holding
        the same three sub-cards as the Automatic review's Shapes step
        (Shape / Outline / Size) with its wording, so one setting reads the
        same in both modes.
        """
        self._refine_panel_title = tr("Outline settings")
        self.refine_group = QWidget()
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        refine_layout = QVBoxLayout(self.refine_group)
        # No gap: open, the header and the card below it are one box, and a gap
        # would cut the box in half (see _apply_refine_toggle).
        refine_layout.setSpacing(0)
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

        # Content card shown/hidden by the header, and drawn joined to it.
        self.refine_content_widget = QWidget()
        self.refine_content_widget.setObjectName("refineContentWidget")
        self.refine_content_widget.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)

        # Shared visible checkbox indicators (off = outlined box, on = blue
        # check): the old inline painter drew the unchecked state as a plain
        # background square, i.e. invisible.
        from .widgets import checkbox_indicator_qss, label_with_target_hint
        _cb_qss = checkbox_indicator_qss(self)

        _refine_qss = _CARD_JOINED_QSS.format(name="refineContentWidget")
        _refine_qss += "QLabel { background: transparent; border: none; }"
        _refine_qss += _cb_qss
        self.refine_content_widget.setStyleSheet(_refine_qss)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        # 9px between sibling sub-cards; each sub-card keeps 6 inside.
        refine_content_layout.setSpacing(9)

        # ── Top level: right angles, round corners (fill holes moved into
        # the collapsed Shape sub-card below) ──

        # Right angles: orthogonalize for man-made shapes. First, as in the
        # Automatic review, because it decides what the rows under it may do.
        ortho_label = _refine_row_label(label_with_target_hint(
            tr("Right angles"), tr("buildings")), tr(
            "Snap walls to right angles, 45 degree walls included. Made for "
            "buildings, pools and solar panels. A shape it would distort is "
            "left as it is."))
        self.right_angles_checkbox = QCheckBox()
        self.right_angles_checkbox.setToolTip(ortho_label.toolTip())
        self.right_angles_checkbox.setChecked(REFINE_ORTHO_DEFAULT)
        # Kept so the availability gate can grey the label with the box
        # (_sync_refine_right_angle_controls).
        self.right_angles_label = ortho_label
        ortho_row = _refine_control_row(ortho_label, self.right_angles_checkbox)

        round_label = _refine_row_label(label_with_target_hint(
            tr("Round corners"), tr("trees")), tr(
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
        self.fill_holes_max_spinbox.setMaximumWidth(scale_px_length(110))
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
        # Two lines, split on the newline (see the same tooltip in
        # auto_review_steps.py): one long line ran the width of the screen.
        points_label = _refine_row_label(tr("Points"), tr(
            "Share of the outline's points to keep. 100% is the standard "
            "density.\nLower thins the smallest detail first, keeps the "
            "corners, and gives Right angles straight walls to square."))
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setSingleStep(5)
        self.points_spinbox.setRange(1, 100)
        self.points_spinbox.setValue(REFINE_POINTS_PCT_DEFAULT)
        self.points_spinbox.setSuffix(" %")
        self.points_spinbox.setMinimumWidth(62)
        self.points_spinbox.setMaximumWidth(scale_px_length(78))
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
        self.simplify_spinbox.setMaximumWidth(scale_px_length(78))
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
        self.clean_edges_spinbox.setMaximumWidth(scale_px_length(78))
        clean_row = _refine_control_row(clean_label, self.clean_edges_spinbox)

        expand_label = _refine_row_label(tr("Grow / shrink"), tr(
            "Positive = grow outward, negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(REFINE_EXPAND_DEFAULT)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setToolTip(expand_label.toolTip())
        self.expand_spinbox.setMinimumWidth(62)
        self.expand_spinbox.setMaximumWidth(scale_px_length(78))
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
        self.min_size_spinbox.setMaximumWidth(scale_px_length(110))
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
        self.max_size_spinbox.setMaximumWidth(scale_px_length(110))
        size_row.addWidget(min_size_label)
        size_row.addWidget(self.min_size_spinbox)
        size_row.addStretch()
        size_row.addWidget(max_size_label)
        size_row.addWidget(self.max_size_spinbox)

        # Assembly: the two toggles users actually reach for sit at top level
        # (usage data: Right angles and Round corners are the touched refine
        # controls), so with the hover ghost live they re-shape the outline
        # without opening anything. The long tail keeps the Automatic review's
        # Shape / Outline / Size cut, inside a collapsed More settings section,
        # so the panel stays a few lines tall by default.
        # One container for the pair, because the offline engine does not get
        # them (_sync_refine_shape_toggles) and a layout cannot be hidden.
        self.refine_shape_toggles = QWidget()
        top_rows = QVBoxLayout(self.refine_shape_toggles)
        top_rows.setContentsMargins(0, 0, 0, 0)
        top_rows.setSpacing(6)
        top_rows.addLayout(ortho_row)
        top_rows.addLayout(round_row)
        refine_content_layout.addWidget(self.refine_shape_toggles)

        # The house collapsible header (chevron + normal-case title), joined
        # to the body it opens exactly like the panel header above it: open, it
        # drops its bottom edge and the card below carries the other three.
        # The pair sits in its own zero-spacing column, because the content
        # layout puts 9px between its children and that gap would cut the box
        # in half.
        more_column = QVBoxLayout()
        more_column.setContentsMargins(0, 0, 0, 0)
        more_column.setSpacing(0)
        self._refine_more_expanded = _read_refine_more_expanded()
        self.refine_more_btn = QPushButton()
        self.refine_more_btn.setStyleSheet(_SECTION_TOGGLE_QSS)
        self.refine_more_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refine_more_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refine_more_btn.clicked.connect(self._on_refine_more_clicked)
        more_column.addWidget(self.refine_more_btn)

        self.refine_more_widget = QWidget()
        self.refine_more_widget.setObjectName("refineMoreWidget")
        self.refine_more_widget.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.refine_more_widget.setStyleSheet(
            _CARD_JOINED_QSS.format(name="refineMoreWidget")
            + "QLabel { background: transparent; border: none; }")
        more_layout = QVBoxLayout(self.refine_more_widget)
        more_layout.setContentsMargins(10, 10, 10, 10)
        # 9px between sibling sub-cards, as before the split.
        more_layout.setSpacing(9)
        more_layout.addWidget(_settings_zone(
            "refineZoneShape", tr("Shape"),
            tr("how the outline is styled"), [
                fill_row,
                self.fill_holes_max_row,
            ]))
        more_layout.addWidget(_settings_zone(
            "refineZoneOutline", tr("Outline"),
            tr("fine-tune the edges"), [
                points_row,
                simplify_row,
                clean_row,
                expand_row,
            ]))
        more_layout.addWidget(_settings_zone(
            "refineZoneSize", tr("Size"),
            tr("hide anything outside this range"), [
                size_row,
            ]))
        more_column.addWidget(self.refine_more_widget)
        refine_content_layout.addLayout(more_column)
        self._apply_refine_more_toggle()

        refine_layout.addWidget(self.refine_content_widget)
        self._apply_refine_toggle(self._refine_expanded)

        # Connect signals
        self.points_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.clean_edges_spinbox.valueChanged.connect(self._on_refine_changed)
        self.round_corners_checkbox.stateChanged.connect(self._on_refine_ticked)
        # The two syncs are connected BEFORE the emit they belong with. A tick
        # used to reach the emit 150 ms later, so the panel had always settled
        # first; now the emit is immediate, and the order is what keeps it
        # reading a settled panel.
        self.right_angles_checkbox.stateChanged.connect(
            self._sync_refine_right_angle_controls)
        self.right_angles_checkbox.stateChanged.connect(self._on_refine_ticked)
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(
            self._sync_fill_holes_max_row)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_ticked)
        self.fill_holes_max_spinbox.valueChanged.connect(self._on_refine_changed)
        self.min_size_spinbox.valueChanged.connect(self._on_refine_changed)
        self.max_size_spinbox.valueChanged.connect(self._on_refine_changed)

        self._sync_refine_right_angle_controls()

        # Last, so it writes over the defaults the syncs above just settled.
        # It blocks the signals it touches, so nothing reshapes an outline that
        # does not exist yet.
        self._apply_refine_start_values()
        self._sync_refine_shape_toggles()
        parent_layout.addWidget(self.refine_group)

    # -- what the panel opens on --------------------------------------------

    def _refine_start_values(self) -> dict:
        """What each remembered control starts on.

        The default in force for a control the user never moved, and their own
        last choice for one they did. Never raises: an unreadable memory just
        leaves the defaults standing.
        """
        return refine_start_values({
            "right_angles": bool(REFINE_ORTHO_DEFAULT),
            "round_corners": REFINE_SMOOTH_DEFAULT > 0,
            "fill_holes": bool(REFINE_FILL_HOLES_DEFAULT),
            "fill_holes_max_m2": float(REFINE_FILL_HOLES_MAX_M2_DEFAULT),
            "points_pct": int(REFINE_POINTS_PCT_DEFAULT),
            "simplify_px": float(REFINE_SIMPLIFY_DEFAULT),
            "clean_px": float(REFINE_CLEAN_DEFAULT),
            "expand_px": int(REFINE_EXPAND_DEFAULT),
        })

    def _apply_refine_start_values(self) -> None:
        """Put the start values on the controls, emitting nothing."""
        start = self._refine_start_values()
        apply_refine_start_values(self, start)
        self._forget_round_corners_memory()
        self._sync_fill_holes_max_row()
        if start.get("right_angles"):
            # The check behind Right angles is a real shapely import, and this
            # panel is built while the plugin loads. A remembered tick must not
            # pull that in there, so it goes to the next turn of the event
            # loop: load stays light, and a machine without the library still
            # hears about it straight away.
            safe_single_shot(0, self, self._sync_refine_right_angle_controls)
        else:
            self._sync_refine_right_angle_controls()

    def publish_refine_settings(self) -> None:
        """Hand the panel's settled values to whoever acts on them.

        The settings exist twice, on the panel and on the side that shapes the
        outline, and the two reset to different baselines: the panel to what
        this user last chose, the pipeline to the shipped defaults. Nothing
        emitted in between, so a ticked Right angles could sit over a squaring
        that was off. Every reset ends here now.

        Guarded against re-entry: a handler that repaints must not come back
        through this. Silent in a handoff, where the panel carries one
        polygon's values and the emit would reshape the open edit.
        """
        if getattr(self, "_publishing_refine_settings", False):
            return
        if getattr(self, "_refine_handoff", False):
            return
        self._publishing_refine_settings = True
        try:
            # A spinbox moved just before the reset leaves the debounce timer
            # armed, and it would fire after this call with the panel already
            # back at its start values: one more emit, from a state nobody
            # asked for, and past the handoff guard above.
            timer = getattr(self, "_refine_debounce_timer", None)
            if timer is not None:
                try:
                    timer.stop()
                except (RuntimeError, AttributeError):
                    pass
            self._emit_refine_changed()
        except (RuntimeError, AttributeError):
            pass  # nosec B110 -- panel torn down mid-publish
        finally:
            self._publishing_refine_settings = False

    def _mark_refine_touched(self) -> None:
        """Note that the user moved the control that just emitted.

        Only a touched control is written to the memory, so a served default
        keeps answering for a setting nobody chose. Every programmatic write
        to these controls blocks its signals, so what reaches here is the
        user's own hand. A handoff is not: it seeds the panel from one
        polygon, and those values are not a preference.
        """
        if getattr(self, "_refine_handoff", False):
            return
        with suppress(RuntimeError, AttributeError, TypeError):
            name = refine_setting_name_for(self, self.sender())
            if not name:
                return
            keys = getattr(self, "_refine_touched_keys", None)
            if keys is None:
                keys = set()
                self._refine_touched_keys = keys
            keys.add(name)

    def _remember_refine_settings(self) -> None:
        """Write the panel's settled state, so the next session opens on it.

        Not in a handoff: the panel is showing one polygon's stored values
        there, seeded from the Automatic review, and remembering those would
        open every later Semi-Auto session on a shape the user never chose.
        """
        if getattr(self, "_refine_handoff", False):
            return
        touched = getattr(self, "_refine_touched_keys", None)
        if not touched:
            return
        settled = capture_refine_settings(self, only=touched)
        # Right angles forces Round corners off and holds the tick that was
        # there. That held tick is the user's own choice, so it is what gets
        # written: squaring an outline used to erase a Round corners
        # preference for every later session.
        held_round = getattr(self, "_round_corners_before_right_angles", None)
        if held_round is not None and "round_corners" in settled:
            settled["round_corners"] = bool(held_round)
        # Offline, the two shape boxes are held off by the panel, not by the
        # user. Their real ticks are in the memory, and that is what gets
        # written: an offline session must not erase a preference.
        for key, memory_name in (
                ("right_angles", "_right_angles_before_offline"),
                ("round_corners", "_round_corners_before_offline")):
            held = getattr(self, memory_name, None)
            if held is not None and key in settled:
                settled[key] = bool(held)
        remember_refine_settings(settled)

    def _sync_fill_holes_max_row(self, _state=None) -> None:
        """Show the size threshold only while Fill holes is on (an irrelevant
        control is hidden, never greyed)."""
        row = getattr(self, "fill_holes_max_row", None)
        if row is not None:
            row.setVisible(self.fill_holes_checkbox.isChecked())

    def _refine_shape_toggles_allowed(self) -> bool:
        """Whether Right angles and Round corners belong on this panel.

        They shape what the cloud model traces, and they are what a Semi-Auto
        user gets for moving to Cloud AI, so an offline session does not carry
        them. Two cases keep them anyway: the per-polygon handoff, which
        reshapes geometry already saved with no engine behind it, and a server
        that offers no second engine at all, where there is nothing to move to.

        Fails OPEN: an unreadable route is not a reason to take a control away
        from someone who has always had it.
        """
        if getattr(self, "_refine_handoff", False):
            return True
        try:
            from ...core.manual_cloud_route import manual_cloud_route_offered

            if not manual_cloud_route_offered():
                return True
            return bool(self._manual_cloud_route_picked())
        except (RuntimeError, AttributeError, ImportError):
            return True

    def _sync_refine_shape_toggles(self) -> None:
        """Show the pair on Cloud AI, hide it offline (hidden, never greyed,
        like every other cloud-only affordance in this panel).

        Hidden, both are forced off, so no pass the user cannot see shapes an
        offline outline. Their ticks are remembered and given back the moment
        the cloud answers again, exactly as Right angles does with Round
        corners.
        """
        rows = getattr(self, "refine_shape_toggles", None)
        if rows is None:
            return
        allowed = self._refine_shape_toggles_allowed()
        rows.setVisible(allowed)
        changed = False
        pairs = (
            ("right_angles_checkbox", "_right_angles_before_offline"),
            ("round_corners_checkbox", "_round_corners_before_offline"),
        )
        for widget_name, memory_name in pairs:
            box = getattr(self, widget_name, None)
            if box is None:
                continue
            if not allowed:
                if getattr(self, memory_name, None) is None:
                    # Right angles may be holding Round corners off right now.
                    # What goes to memory is the user's own tick, which in that
                    # case is the one Right angles took.
                    held = getattr(
                        self, "_round_corners_before_right_angles", None)
                    setattr(self, memory_name, box.isChecked() if (
                        memory_name != "_round_corners_before_offline"
                        or held is None) else bool(held))
                wanted = False
            else:
                remembered = getattr(self, memory_name, None)
                if remembered is None:
                    continue
                setattr(self, memory_name, None)
                wanted = bool(remembered)
            try:
                if bool(box.isChecked()) != bool(wanted):
                    changed = True
                box.blockSignals(True)
                box.setChecked(wanted)
            except (RuntimeError, AttributeError):
                pass
            finally:
                try:
                    box.blockSignals(False)
                except (RuntimeError, AttributeError):
                    pass
        self._sync_refine_right_angle_controls()
        # Every setChecked above ran with the signals blocked, so the side that
        # shapes the outline heard none of them. Say it once, and only when
        # something actually moved.
        if changed:
            self.publish_refine_settings()

    def _sync_refine_right_angle_controls(self, _state=None) -> None:
        """Make incompatible Shape controls unavailable with Right angles.

        The Automatic review's rule (_sync_auto_right_angle_controls), applied
        to the Manual panel. _emit_refine_changed enforces the same thing on the
        values, so a disabled widget can never leave an old setting active.

        Right angles itself is refused here when the geometry library behind it
        is missing (right_angles_support), on the same terms as the review: only
        a TICKED box is tested, and the seeded default is off, so the
        build-time call never imports shapely at plugin load.
        """
        ortho = getattr(self, "right_angles_checkbox", None)
        if ortho is not None and ortho.isChecked():
            from .right_angles_support import gate_right_angles

            gate_right_angles(ortho, getattr(self, "right_angles_label", None))
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
        # never leaves a hidden rounding pass in the preview. The user's own
        # choice is remembered while it is forced off, and given back the
        # moment Right angles lets go, so a tick they made is never lost to a
        # control they only turned on and off again.
        round_corners = getattr(self, "round_corners_checkbox", None)
        if round_corners is not None:
            if not enabled:
                if getattr(self, "_round_corners_before_right_angles", None) is None:
                    self._round_corners_before_right_angles = round_corners.isChecked()
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
            else:
                remembered = getattr(
                    self, "_round_corners_before_right_angles", None)
                if remembered is not None:
                    self._round_corners_before_right_angles = None
                    try:
                        round_corners.blockSignals(True)
                        round_corners.setChecked(remembered)
                    except (RuntimeError, AttributeError):
                        pass
                    finally:
                        try:
                            round_corners.blockSignals(False)
                        except (RuntimeError, AttributeError):
                            pass

    def _forget_refine_engine_memory(self) -> None:
        """Drop the ticks held while the offline engine hides the pair.

        They belong to the session they were taken from. A reset puts new
        values on those boxes, and handing the old ones back at the next
        switch to Cloud AI would quietly overwrite them.
        """
        self._right_angles_before_offline = None
        self._round_corners_before_offline = None

    def _forget_round_corners_memory(self) -> None:
        """Drop the Round corners tick held while Right angles forces it off.

        Every setter that writes the two controls explicitly calls this first.
        The remembered tick belongs to the state it was taken from, and handing
        it back over a value the caller just set would overwrite the caller.
        """
        self._round_corners_before_right_angles = None

    def _refresh_refine_more_header(self) -> None:
        """Chevron + normal-case title on the More settings header."""
        arrow = "▾" if self._refine_more_expanded else "▸"
        self.refine_more_btn.setText(arrow + " " + tr("More settings"))

    def _on_refine_more_clicked(self) -> None:
        """More settings clicked: flip the long tail. Pure setVisible.

        The choice is remembered across sessions: a user who works with the
        long tail open should not reopen it at every plugin start.
        """
        self._refine_more_expanded = not self._refine_more_expanded
        _write_refine_more_expanded(self._refine_more_expanded)
        self._apply_refine_more_toggle()

    def _apply_refine_more_toggle(self) -> None:
        """Show/hide the long-tail card, swap the header style and sync the
        chevron. Open, header and card draw as one box."""
        self.refine_more_widget.setVisible(self._refine_more_expanded)
        self.refine_more_btn.setStyleSheet(
            _SECTION_TOGGLE_OPEN_QSS if self._refine_more_expanded
            else _SECTION_TOGGLE_QSS)
        self._refresh_refine_more_header()

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
        """Show/hide the content card and sync the header chevron.

        The header swaps stylesheet with it. Open, it drops its bottom edge and
        its bottom corners so it and the card draw as a single box: the settings
        ARE what this header opens, and two separate cards said otherwise.
        """
        self.refine_content_widget.setVisible(expanded)
        self.refine_toggle_btn.setStyleSheet(
            _SECTION_TOGGLE_OPEN_QSS if expanded else _SECTION_TOGGLE_QSS)
        self._refresh_refine_header()

    def _on_refine_changed(self, value=None):
        """A refine NUMBER moved: wait out the rest of the run of values.

        A spinbox held down, or typed into, emits once per step, and the
        preview behind it re-shapes the whole outline. The wait absorbs the run
        without letting the shape lag behind the number for longer than it
        takes to read it.
        """
        self._mark_refine_touched()
        self._refine_debounce_timer.start(refine_settle_ms())

    def _on_refine_ticked(self, _state=None):
        """A refine BOX was ticked: apply it now.

        A checkbox has no run of values to wait out. It emits once, the user is
        looking straight at the outline, and making them wait was the whole of
        the delay they felt on Round corners, Right angles and Fill holes.
        """
        self._mark_refine_touched()
        self._refine_debounce_timer.stop()
        self._emit_refine_changed()

    def _emit_refine_changed(self):
        """Emit the refine settings changed signals after debounce.

        The size filter, the fill-holes threshold and the outline budget go out
        FIRST (their handlers only store the values), so the
        refine_settings_changed handler that follows repaints once with
        everything fresh.
        """
        # Offline, the pair is not on the panel at all, so its values are not
        # the user's to give: read them as off whatever the boxes hold.
        shape_allowed = self._refine_shape_toggles_allowed()
        right_angles = shape_allowed and self.right_angles_checkbox.isChecked()
        # Written from the settled panel, which is the user's own choice and
        # never a per-polygon value: set_refine_values blocks the signals that
        # reach here, so restoring a saved shape cannot overwrite the memory.
        self._remember_refine_settings()
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
            0 if right_angles or not shape_allowed else (
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
        """Put the panel back where a new session starts, emitting nothing.

        Back to THIS user's start values, not to the shipped ones: a reset
        that undid the settings they chose for the objects they work on would
        make the memory useless every time a session ended. The size filters
        are read against one dataset and are not remembered, so they go back
        to off.
        """
        start = self._refine_start_values()
        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox,
                  self.min_size_spinbox, self.max_size_spinbox):
            w.blockSignals(True)

        self.points_spinbox.setValue(int(start["points_pct"]))
        self.simplify_spinbox.setValue(float(start["simplify_px"]))
        self.clean_edges_spinbox.setValue(float(start["clean_px"]))
        self.round_corners_checkbox.setChecked(bool(start["round_corners"]))
        self.right_angles_checkbox.setChecked(bool(start["right_angles"]))
        self.expand_spinbox.setValue(int(start["expand_px"]))
        self.fill_holes_checkbox.setChecked(bool(start["fill_holes"]))
        self.fill_holes_max_spinbox.setValue(float(start["fill_holes_max_m2"]))
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
        self._forget_round_corners_memory()
        self._forget_refine_engine_memory()
        self._sync_fill_holes_max_row()
        self._sync_refine_right_angle_controls()
        self._sync_refine_shape_toggles()

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
        self._forget_round_corners_memory()
        self._forget_refine_engine_memory()
        self._sync_fill_holes_max_row()
        self._sync_refine_right_angle_controls()
        self._sync_refine_shape_toggles()

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
