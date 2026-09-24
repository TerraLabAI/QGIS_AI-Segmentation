






from __future__ import annotations

from qgis.PyQt.QtCore import Qt
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
    AUTO_REVIEW_FILL_HOLES_MAX_M2_DEFAULT as _AUTO_REVIEW_FILL_MAX_M2_DEFAULT,
)
from ...core.review_defaults import (
    AUTO_REVIEW_POINTS_PCT_DEFAULT as _AUTO_REVIEW_POINTS_PCT_DEFAULT,
)
from ...core.review_defaults import auto_review_clean_default as _auto_review_clean_default
from ...core.review_defaults import auto_review_expand_default as _auto_review_expand_default
from ...core.review_defaults import (
    auto_review_fill_holes_default as _auto_review_fill_holes_default,
)
from ...core.review_defaults import auto_review_ortho_default as _auto_review_ortho_default
from ...core.review_defaults import auto_review_simplify_default as _auto_review_simplify_default
from ...core.review_defaults import auto_review_smooth_default as _auto_review_smooth_default
from ...core.shape_policy_dials import auto_review_points_pct_default
from .auto_correct_build import _review_step_heading
from .fold_row import FoldRow
from .font_scale import fit_spin_width
from .review_card_rows import (
    REVIEW_LIST_MARGINS,
    review_list_row,
    review_zone,
    row_lead_glyph,
    row_title_with_note,
)


def _icon_setting_row(icon_name: str, label, control,
                      first: bool = False) -> QWidget:





    row, lay = review_list_row(first=first)


    lay.addWidget(row_lead_glyph(icon_name), 0,
                  Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    lay.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
    lay.addStretch(1)
    lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
    return row


def _shapes_card(obj_name: str) -> tuple[QWidget, QVBoxLayout]:

    card, col = review_zone(obj_name)
    col.setContentsMargins(*REVIEW_LIST_MARGINS)
    col.setSpacing(0)
    return card, col


def build_shapes_page(dock) -> QWidget:



    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(0, 6, 0, 0)
    lay.setSpacing(6)

    dock.auto_shapes_heading = _review_step_heading(
        tr("Clean up the outlines"),
        tr("applies to every polygon"))
    lay.addWidget(dock.auto_shapes_heading)


    dock.auto_shape_content = QWidget()
    _shape_layout = QVBoxLayout(dock.auto_shape_content)
    _shape_layout.setContentsMargins(0, 2, 0, 0)

    _shape_layout.setSpacing(8)


















    _points_lbl = QLabel(tr("Points"))
    _points_lbl.setObjectName("autoFieldLabel")


    _points_lbl.setToolTip(tr(
        "Share of each outline's points to keep. 100% is the class default.\n"
        "Lower thins the smallest detail first while keeping the corners."))
    dock.auto_points_spin = QSpinBox()
    dock.auto_points_spin.setSingleStep(5)



    dock.auto_points_spin.setRange(1, 100)
    dock.auto_points_spin.setValue(
        auto_review_points_pct_default(_AUTO_REVIEW_POINTS_PCT_DEFAULT))
    dock.auto_points_spin.setSuffix(" %")
    fit_spin_width(dock.auto_points_spin, 84, 96)
    dock.auto_points_spin.setToolTip(_points_lbl.toolTip())
    _points_hdr = _icon_setting_row("points", _points_lbl, dock.auto_points_spin)






    _simplify_lbl = QLabel(tr("Simplify"))
    _simplify_lbl.setObjectName("autoFieldLabel")
    _simplify_lbl.setToolTip(tr(
        "Drop points closer than this distance to a straight edge (0 = off)."))
    dock.auto_simplify_spin = QDoubleSpinBox()
    dock.auto_simplify_spin.setDecimals(1)
    dock.auto_simplify_spin.setSingleStep(0.5)
    dock.auto_simplify_spin.setRange(0.0, 1000.0)
    dock.auto_simplify_spin.setValue(_auto_review_simplify_default())
    dock.auto_simplify_spin.setSuffix(" px")
    fit_spin_width(dock.auto_simplify_spin, 84, 96)
    dock.auto_simplify_spin.setToolTip(_simplify_lbl.toolTip())
    _simplify_hdr = _icon_setting_row(
        "simplify", _simplify_lbl, dock.auto_simplify_spin)






    _clean_lbl = QLabel(tr("Trim spikes"))
    _clean_lbl.setObjectName("autoFieldLabel")
    _clean_lbl.setToolTip(tr(
        "Shave thin spikes and ragged bits off each shape's outline. It leaves "
        "the main body alone; higher values trim more. 0 = off."))
    dock.auto_clean_spin = QDoubleSpinBox()
    dock.auto_clean_spin.setDecimals(1)
    dock.auto_clean_spin.setSingleStep(0.5)
    dock.auto_clean_spin.setRange(0.0, 50.0)
    dock.auto_clean_spin.setValue(_auto_review_clean_default())
    dock.auto_clean_spin.setSuffix(" px")
    fit_spin_width(dock.auto_clean_spin, 84, 96)
    dock.auto_clean_spin.setToolTip(_clean_lbl.toolTip())
    _clean_hdr = _icon_setting_row("trim", _clean_lbl, dock.auto_clean_spin)



    _round_lbl = QLabel(tr("Round corners"))
    _round_lbl.setObjectName("autoFieldLabel")
    _round_lbl.setToolTip(tr(
        "Round corners for natural shapes like trees and bushes. "
        "Lower Points for smoother results."))
    dock.auto_round_corners_check = QCheckBox()
    dock.auto_round_corners_check.setChecked(_auto_review_smooth_default())
    dock.auto_round_corners_check.setToolTip(_round_lbl.toolTip())
    _round_row = _icon_setting_row(
        "round_corner", row_title_with_note(_round_lbl, tr("trees")),
        dock.auto_round_corners_check)




    _ortho_lbl = QLabel(tr("Right angles"))
    _ortho_lbl.setObjectName("autoFieldLabel")
    _ortho_lbl.setToolTip(tr(
        "Snap walls to right angles, 45 degree walls included. Made for "
        "buildings, pools and solar panels. A shape it would distort is "
        "left as it is."))
    dock.auto_ortho_check = QCheckBox()
    dock.auto_ortho_check.setChecked(_auto_review_ortho_default())
    dock.auto_ortho_check.setToolTip(_ortho_lbl.toolTip())


    dock.auto_ortho_label = _ortho_lbl
    _ortho_row = _icon_setting_row(
        "right_angle", row_title_with_note(_ortho_lbl, tr("buildings")),
        dock.auto_ortho_check, first=True)




    dock.auto_ortho_unavailable_label = QLabel("")
    dock.auto_ortho_unavailable_label.setWordWrap(True)
    dock.auto_ortho_unavailable_label.setObjectName("autoHint")
    dock.auto_ortho_unavailable_label.setVisible(False)







    dock._auto_right_angle_conflicts = (
        _clean_lbl,
        dock.auto_clean_spin,
        _round_lbl,
        dock.auto_round_corners_check,
    )
    dock._auto_right_angle_conflict_tooltips = tuple(
        (widget, widget.toolTip()) for widget in dock._auto_right_angle_conflicts)

    _expand_lbl = QLabel(tr("Grow / shrink"))
    _expand_lbl.setObjectName("autoFieldLabel")
    _expand_lbl.setToolTip(tr(
        "Positive = grow outward, negative = shrink inward"))
    dock.auto_expand_spin = QSpinBox()
    dock.auto_expand_spin.setRange(-1000, 1000)
    dock.auto_expand_spin.setValue(_auto_review_expand_default())
    dock.auto_expand_spin.setSuffix(" px")
    fit_spin_width(dock.auto_expand_spin, 84, 96)


    dock.auto_expand_spin.setToolTip(_expand_lbl.toolTip())
    _expand_row = _icon_setting_row(
        "grow_shrink", _expand_lbl, dock.auto_expand_spin)





    _fill_lbl = QLabel(tr("Fill holes"))
    _fill_lbl.setObjectName("autoFieldLabel")
    _fill_lbl.setToolTip(tr("Fill interior holes in the selection"))
    dock.auto_fill_holes_check = QCheckBox()
    dock.auto_fill_holes_check.setChecked(_auto_review_fill_holes_default())
    dock.auto_fill_holes_check.setToolTip(_fill_lbl.toolTip())
    _fill_row = _icon_setting_row(
        "fill_holes", _fill_lbl, dock.auto_fill_holes_check)



    dock.auto_fill_max_row = QWidget()
    _fill_child = QVBoxLayout(dock.auto_fill_max_row)


    _fill_child.setContentsMargins(26, 0, 0, 8)
    _fill_child.setSpacing(0)
    _fill_max_row = QHBoxLayout()
    _fill_max_row.setContentsMargins(0, 0, 0, 0)
    _fill_max_row.setSpacing(6)
    _fill_max_lbl = QLabel(tr("Up to"))
    _fill_max_lbl.setObjectName("autoHint")
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
    fit_spin_width(dock.auto_fill_max_spin, 78, 110)
    _fill_max_row.addWidget(_fill_max_lbl)
    _fill_max_row.addStretch()
    _fill_max_row.addWidget(dock.auto_fill_max_spin)
    _fill_child.addLayout(_fill_max_row)
    dock.auto_fill_max_row.setVisible(dock.auto_fill_holes_check.isChecked())



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











    _switches, _sw_col = _shapes_card("autoShapeSwitches")

    dock.auto_ortho_unavailable_label.setContentsMargins(26, 0, 0, 8)


    dock.auto_round_corners_row = _round_row
    _fill_block = QWidget()
    _fill_col = QVBoxLayout(_fill_block)
    _fill_col.setContentsMargins(0, 0, 0, 0)
    _fill_col.setSpacing(0)
    _fill_col.addWidget(_fill_row)
    _fill_col.addWidget(dock.auto_fill_max_row)
    for _item in (_ortho_row, dock.auto_ortho_unavailable_label,
                  dock.auto_round_corners_row, _fill_block):
        _sw_col.addWidget(_item)
    _shape_layout.addWidget(_switches)

    _more, _more_col = _shapes_card("autoShapeMoreZone")

    dock.auto_shapes_more_btn = FoldRow(
        tr("More settings"),
        settings_key="AISegmentation/review/shapes_more_expanded")





    dock.auto_shapes_more_btn.set_fold_fact(
        tr("Points, Simplify, Trim, Grow"))
    _more_col.addWidget(dock.auto_shapes_more_btn)
    dock.auto_shapes_more_body = QWidget()
    _body_col = QVBoxLayout(dock.auto_shapes_more_body)
    _body_col.setContentsMargins(0, 0, 0, 0)
    _body_col.setSpacing(0)
    for _row in (_points_hdr, _simplify_hdr, _clean_hdr, _expand_row):
        _body_col.addWidget(_row)
    _more_col.addWidget(dock.auto_shapes_more_body)
    dock.auto_shapes_more_btn.bind_fold_body(dock.auto_shapes_more_body)
    _shape_layout.addWidget(_more)

    lay.addWidget(dock.auto_shape_content)

    lay.addStretch(1)
    dock._sync_auto_right_angle_controls()
    return page
