"""Automatic review step pages: the Shapes page and the hand-edit block.

Module-level builders (not a mixin, so the dock assembly keeps its shape):
each takes the dock instance and builds one part of the review card's step
stack (see auto_review_build.py). Dumb widgets only; auto_state.py holds the
store-only setters that drive them.
"""
from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
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
from .auto_correct_build import _review_step_heading
from .styles import _settings_zone


def build_shapes_page(dock) -> QWidget:
    """Shapes page (step 3, last): the Shape / Outline controls, each its own
    sub-card. Nothing here removes an object: the filters that do live on the
    Keep step."""
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)

    dock.auto_shapes_heading = _review_step_heading(
        tr("Clean up the outlines"),
        tr("applies to every polygon"))
    lay.addWidget(dock.auto_shapes_heading)

    # The shape / outline controls ARE this step. They group into two sub-cards
    # (Shape / Outline, see styles._settings_zone) so the step reads as panels,
    # not one flat block.
    dock.auto_shape_content = QWidget()
    _shape_layout = QVBoxLayout(dock.auto_shape_content)
    _shape_layout.setContentsMargins(0, 2, 0, 0)
    # 9px between sibling sub-cards; each sub-card keeps 6 inside.
    _shape_layout.setSpacing(9)

    # Shape refinement (points / round corners / expand-shrink) acts on the
    # WHOLE detected objects, live and free. They mirror the Manual refine
    # knobs, applied to the merged geometry (there is no mask left in review).
    # Any change re-derives the visible set via auto_refine_changed (debounced,
    # cooperative).

    # Points: the share of its own points each outline keeps. A mask traced
    # pixel by pixel carries one point every few centimetres, which is the
    # raster grid showing through, not accuracy. Lowering this drops the
    # smallest detail first and refits the surviving edges onto the middle of
    # what they replace, so the outline thins without shrinking.
    #
    # It is a COUNT, not a distance. The distance tolerance this row used to
    # hold ("Simplify", px) reduced the count only as a side effect: to reach a
    # sane number of points it had to allow a deviation large enough to sweep
    # whole walls into one slanted chord, which is what left buildings with
    # triangular corners and nothing square for Right angles to snap.
    _points_hdr = QHBoxLayout()
    _points_lbl = QLabel(tr("Points"))
    _points_lbl.setStyleSheet("font-size: 11px;")
    _points_lbl.setToolTip(tr(
        "How many of each outline's points to keep, as a share. 100% keeps the "
        "class default; lower thins further, dropping the smallest detail "
        "first and keeping the corners and the surface. With Right angles on "
        "it runs first, so lowering it gives the squaring straight walls to "
        "work from."))
    _points_hdr.addWidget(_points_lbl)
    _points_hdr.addStretch()
    dock.auto_points_spin = QSpinBox()
    dock.auto_points_spin.setSingleStep(5)
    # Down to 1%: on a class with no spacing the budget is a share of the raw
    # traced points, so a dense outline (hundreds of points) still needs a very
    # low share to reach a hand-drawn count. Stepping from 5 reaches 1.
    dock.auto_points_spin.setRange(1, 100)
    dock.auto_points_spin.setValue(_AUTO_REVIEW_POINTS_PCT_DEFAULT)
    dock.auto_points_spin.setSuffix(" %")
    dock.auto_points_spin.setMinimumWidth(62)
    dock.auto_points_spin.setMaximumWidth(78)
    dock.auto_points_spin.setToolTip(_points_lbl.toolTip())
    _points_hdr.addWidget(dock.auto_points_spin)

    # Simplify (px): the older Douglas-Peucker tolerance, kept alongside Points
    # so the two reducers can be compared on the same outline. It is a distance,
    # not a count: it removes any point closer than this to the chord that would
    # replace it. 0 = off. Pushed hard it produces the slanted-chord look Points
    # avoids, which is exactly why both are here.
    _simplify_hdr = QHBoxLayout()
    _simplify_lbl = QLabel(tr("Simplify"))
    _simplify_lbl.setStyleSheet("font-size: 11px;")
    _simplify_lbl.setToolTip(tr(
        "Drop points closer than this distance to a straight edge (0 = off). "
        "A distance, not a count: pushed high it can flatten curved walls. "
        "Points is usually the better dial; this stays for comparison."))
    _simplify_hdr.addWidget(_simplify_lbl)
    _simplify_hdr.addStretch()
    dock.auto_simplify_spin = QDoubleSpinBox()
    dock.auto_simplify_spin.setDecimals(1)
    dock.auto_simplify_spin.setSingleStep(0.5)
    dock.auto_simplify_spin.setRange(0.0, 1000.0)
    dock.auto_simplify_spin.setValue(_AUTO_REVIEW_SIMPLIFY_DEFAULT)
    dock.auto_simplify_spin.setSuffix(" px")
    dock.auto_simplify_spin.setMinimumWidth(62)
    dock.auto_simplify_spin.setMaximumWidth(78)
    dock.auto_simplify_spin.setToolTip(_simplify_lbl.toolTip())
    _simplify_hdr.addWidget(dock.auto_simplify_spin)

    # Clean edges: morphological opening (shrink-then-grow) that strips thin
    # attached fringe / tendrils left around objects. Unlike Min size
    # (which only drops SEPARATE small features), this removes noise that is
    # part of the SAME polygon. Light default so panels/buildings come out
    # clean; 0 = off. px, converted to ground units by the source pixel size.
    _clean_hdr = QHBoxLayout()
    _clean_lbl = QLabel(tr("Trim spikes"))
    _clean_lbl.setStyleSheet("font-size: 11px;")
    _clean_lbl.setToolTip(tr(
        "Shave thin spikes and ragged bits off each shape's outline. It leaves "
        "the main body alone; higher values trim more. 0 = off."))
    _clean_hdr.addWidget(_clean_lbl)
    _clean_hdr.addStretch()
    dock.auto_clean_spin = QDoubleSpinBox()
    dock.auto_clean_spin.setDecimals(1)
    dock.auto_clean_spin.setSingleStep(0.5)
    dock.auto_clean_spin.setRange(0.0, 50.0)
    dock.auto_clean_spin.setValue(_AUTO_REVIEW_CLEAN_DEFAULT)
    dock.auto_clean_spin.setSuffix(" px")
    dock.auto_clean_spin.setMinimumWidth(62)
    dock.auto_clean_spin.setMaximumWidth(78)
    dock.auto_clean_spin.setToolTip(_clean_lbl.toolTip())
    _clean_hdr.addWidget(dock.auto_clean_spin)

    # Round corners + Right angles toggles, and the Grow / shrink spinbox: one
    # control per row, each landing on the right-aligned rail.
    _round_row = QHBoxLayout()
    _round_lbl = QLabel(tr("Round corners"))
    _round_lbl.setStyleSheet("font-size: 11px;")
    _round_lbl.setToolTip(tr(
        "Round corners for natural shapes like trees and bushes. "
        "Lower Points for smoother results."))
    dock.auto_round_corners_check = QCheckBox()
    dock.auto_round_corners_check.setChecked(_AUTO_REVIEW_SMOOTH_DEFAULT)
    dock.auto_round_corners_check.setToolTip(_round_lbl.toolTip())
    _round_row.addWidget(_round_lbl)
    _round_row.addStretch()
    _round_row.addWidget(dock.auto_round_corners_check)

    # Right angles: opt-in orthogonalization for man-made shapes. Applied to
    # ALL visible objects when checked (an explicit user intent beats an
    # automatic rectangularity gate, which would skip L/U-shaped buildings).
    _ortho_row = QHBoxLayout()
    _ortho_lbl = QLabel(tr("Right angles"))
    _ortho_lbl.setStyleSheet("font-size: 11px;")
    _ortho_lbl.setToolTip(tr(
        "Snap walls to right angles, 45 degree walls included. Made for "
        "buildings, pools and solar panels. A shape it would distort is "
        "left as it is."))
    dock.auto_ortho_check = QCheckBox()
    dock.auto_ortho_check.setChecked(_AUTO_REVIEW_ORTHO_DEFAULT)
    dock.auto_ortho_check.setToolTip(_ortho_lbl.toolTip())
    _ortho_row.addWidget(_ortho_lbl)
    _ortho_row.addStretch()
    _ortho_row.addWidget(dock.auto_ortho_check)

    # Right angles is a guided geometry mode. Keep incompatible generic edge
    # controls visible but disabled, and enforce the same rule in the getter.
    # Points is NOT one of them: the squaring runs on a de-staircased outline,
    # and the point budget is the pass that produces one. Locking it left a
    # complex roof stuck with every traced point going into the regularizer,
    # which is what makes a wall come back as a staircase.
    dock._auto_right_angle_conflicts = (
        _clean_lbl,
        dock.auto_clean_spin,
        _round_lbl,
        dock.auto_round_corners_check,
    )
    dock._auto_right_angle_conflict_tooltips = tuple(
        (widget, widget.toolTip()) for widget in dock._auto_right_angle_conflicts)

    _expand_row = QHBoxLayout()
    _expand_lbl = QLabel(tr("Grow / shrink"))
    _expand_lbl.setStyleSheet("font-size: 11px;")
    _expand_lbl.setToolTip(tr(
        "Positive = grow outward, negative = shrink inward"))
    dock.auto_expand_spin = QSpinBox()
    dock.auto_expand_spin.setRange(-1000, 1000)
    dock.auto_expand_spin.setValue(_AUTO_REVIEW_EXPAND_DEFAULT)
    dock.auto_expand_spin.setSuffix(" px")
    dock.auto_expand_spin.setMinimumWidth(62)
    dock.auto_expand_spin.setMaximumWidth(78)
    _expand_row.addWidget(_expand_lbl)
    _expand_row.addStretch()
    _expand_row.addWidget(dock.auto_expand_spin)

    # Fill holes: ON by default on every run, because what survives the
    # per-tile noise pass is still mostly pepper. The "Up to" child below is
    # what protects a real hole (a courtyard, a clearing), not this tick, which
    # the user unticks when an object owns every gap it has.
    _fill_row = QHBoxLayout()
    _fill_lbl = QLabel(tr("Fill holes"))
    _fill_lbl.setStyleSheet("font-size: 11px;")
    _fill_lbl.setToolTip(tr("Fill interior holes in the selection"))
    dock.auto_fill_holes_check = QCheckBox()
    dock.auto_fill_holes_check.setChecked(_AUTO_REVIEW_FILL_HOLES_DEFAULT)
    dock.auto_fill_holes_check.setToolTip(_fill_lbl.toolTip())
    _fill_row.addWidget(_fill_lbl)
    _fill_row.addStretch()
    _fill_row.addWidget(dock.auto_fill_holes_check)

    # Keep the optional threshold as one compact child row.
    dock.auto_fill_max_row = QWidget()
    _fill_child = QVBoxLayout(dock.auto_fill_max_row)
    _fill_child.setContentsMargins(16, 2, 0, 0)
    _fill_child.setSpacing(0)
    _fill_max_row = QHBoxLayout()
    _fill_max_row.setContentsMargins(0, 0, 0, 0)
    _fill_max_row.setSpacing(6)
    _fill_max_lbl = QLabel(tr("Up to"))
    _fill_max_lbl.setStyleSheet("font-size: 11px; color: rgba(128, 128, 128, 0.95);")
    _fill_max_lbl.setToolTip(tr(
        "Fill only holes smaller than this ground area. Bigger holes (a road "
        "median, a courtyard) stay open. No limit = fill every hole."))
    dock.auto_fill_max_spin = QDoubleSpinBox()
    dock.auto_fill_max_spin.setRange(0.0, 1_000_000.0)
    dock.auto_fill_max_spin.setDecimals(1)
    dock.auto_fill_max_spin.setValue(_AUTO_REVIEW_FILL_MAX_M2_DEFAULT)
    dock.auto_fill_max_spin.setSuffix(" m²")
    dock.auto_fill_max_spin.setSpecialValueText(tr("No limit"))
    dock.auto_fill_max_spin.setToolTip(_fill_max_lbl.toolTip())
    dock.auto_fill_max_spin.setMinimumWidth(78)
    dock.auto_fill_max_spin.setMaximumWidth(110)
    _fill_max_row.addWidget(_fill_max_lbl)
    _fill_max_row.addStretch()
    _fill_max_row.addWidget(dock.auto_fill_max_spin)
    _fill_child.addLayout(_fill_max_row)
    dock.auto_fill_max_row.setVisible(_AUTO_REVIEW_FILL_HOLES_DEFAULT)

    # Re-derive the visible set (via the debounced auto_refine_changed) on
    # any shape change; track the adjustment once per control per review.
    dock.auto_points_spin.valueChanged.connect(
        lambda v: dock._on_shape_control_changed("points", v))
    dock.auto_simplify_spin.valueChanged.connect(
        lambda v: dock._on_shape_control_changed("simplify", v))
    dock.auto_clean_spin.valueChanged.connect(
        lambda v: dock._on_shape_control_changed("clean", v))
    dock.auto_round_corners_check.stateChanged.connect(
        lambda s: dock._on_shape_control_changed("round_corners", s))
    dock.auto_ortho_check.stateChanged.connect(
        lambda s: dock._on_shape_control_changed("right_angles", s))
    dock.auto_ortho_check.stateChanged.connect(
        lambda _s: dock._sync_auto_right_angle_controls())
    dock.auto_expand_spin.valueChanged.connect(
        lambda v: dock._on_shape_control_changed("expand", v))
    dock.auto_fill_holes_check.stateChanged.connect(
        lambda s: dock._on_shape_control_changed("fill_holes", s))
    dock.auto_fill_holes_check.stateChanged.connect(
        lambda _s: dock._sync_auto_fill_max_row())
    dock.auto_fill_max_spin.valueChanged.connect(
        lambda v: dock._on_shape_control_changed("fill_holes_size", v))

    # The Min / Max size filters used to sit here, in a third sub-card. They
    # decide which objects survive, not what an outline looks like, so they now
    # live on the Keep step next to Confidence (auto_review_build.py). This step
    # is outlines only.

    # Assembly: two sub-cards, so the step reads as Shape / Outline instead of
    # one flat block of rows. Each header names its group and glosses its job.
    # Shape leads (the headline style decisions) and the numeric outline
    # fine-tuning closes the step. The optional hole threshold stays inside the
    # Shape card as a child of Fill holes.
    _shape_layout.addWidget(_settings_zone(
        "autoShapeZoneShape", tr("Shape"),
        tr("how each outline is styled"), [
            _ortho_row,
            _round_row,
            _fill_row,
            dock.auto_fill_max_row,
        ]))
    _shape_layout.addWidget(_settings_zone(
        "autoShapeZoneOutline", tr("Outline"),
        tr("fine-tune the edges"), [
            _points_hdr,
            _simplify_hdr,
            _clean_hdr,
            _expand_row,
        ]))

    lay.addWidget(dock.auto_shape_content)

    lay.addStretch(1)
    dock._sync_auto_right_angle_controls()
    return page
