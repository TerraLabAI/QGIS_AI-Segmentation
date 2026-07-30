







from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from ..icons import pixmap_for
from .auto_correct_build import _REVIEW_HEADING_QSS
from .auto_flow_look import token_qcolor
from .font_scale import fit_spin_width
from .guidance import BLUE_TINT, HINT_REVIEW_SHARED_BORDERS, DismissibleHint
from .styles import (
    _BTN_LINK_MUTED,
    _CARD_MARGINS,
    INK,
    LINE,
    RADIUS_CARD,
    SURFACE,
)


def review_zone(obj_name: str) -> tuple[QWidget, QVBoxLayout]:








    zone = QWidget()
    zone.setObjectName(obj_name)
    zone.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    zone.setStyleSheet(review_zone_qss(obj_name))
    col = QVBoxLayout(zone)
    col.setContentsMargins(*_CARD_MARGINS)
    col.setSpacing(6)
    return zone, col


def review_zone_qss(obj_name: str) -> str:

    return (f"QWidget#{obj_name} {{ background: {SURFACE};"
            f" border: 1px solid {LINE}; border-radius: {RADIUS_CARD}px; }}"
            "QLabel { background: transparent; border: none; }"
            "QWidget#reviewListRow { background: transparent; border: none;"
            f" border-top: 1px solid {LINE}; border-radius: 0px; }}"
            'QWidget#reviewListRow[first="true"] { border-top: none; }')


def review_list_row(first: bool = False) -> tuple[QWidget, QHBoxLayout]:




    row = QWidget()
    row.setObjectName("reviewListRow")
    row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    row.setProperty("first", bool(first))
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 7, 0, 7)
    lay.setSpacing(10)
    return row, lay



REVIEW_LIST_MARGINS = (12, 3, 12, 3)


def row_title_with_note(title: QLabel, note: str) -> QWidget:



    box = QWidget()
    col = QVBoxLayout(box)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(1)
    col.addWidget(title)
    if note:


        sub = QLabel(note[:1].upper() + note[1:])
        sub.setObjectName("autoHint")
        col.addWidget(sub)
        box.note_label = sub
    return box


def row_lead_glyph(icon_name: str) -> QLabel:

    glyph = QLabel()
    glyph.setPixmap(pixmap_for(glyph, icon_name, 16, token_qcolor(INK)))
    glyph.setFixedSize(16, 16)
    glyph.setStyleSheet("background: transparent; border: none;")
    return glyph


def bounded_size_filter(minimum: float, maximum: float,
                        changed: str) -> tuple[float, float]:





    minimum = float(minimum)
    maximum = float(maximum)
    if maximum <= 0.0 or minimum <= maximum:
        return minimum, maximum
    if changed == "maximum":
        return maximum, maximum
    return minimum, minimum


def build_size_filter_block(dock) -> QWidget:














    block, col = review_zone("autoKeepSizeZone")


    col.setContentsMargins(_CARD_MARGINS[0], _CARD_MARGINS[1],
                           _CARD_MARGINS[2], REVIEW_LIST_MARGINS[3])
    col.setSpacing(2)
    hdr = QHBoxLayout()
    hdr.setSpacing(8)
    hdr.addWidget(row_lead_glyph("size_range"), 0, Qt.AlignmentFlag.AlignVCenter)
    size_lbl = QLabel(tr("Size range to keep"))
    size_lbl.setStyleSheet(_REVIEW_HEADING_QSS)



    note = tr("objects outside it are hidden")
    size_lbl.setToolTip(note[:1].upper() + note[1:] if note else note)
    hdr.addWidget(size_lbl)
    hdr.addStretch(1)
    col.addLayout(hdr)

    min_lbl = QLabel(tr("Minimum"))
    min_lbl.setObjectName("autoFieldLabel")
    dock.auto_min_size_spin = QDoubleSpinBox()
    dock.auto_min_size_spin.setRange(0.0, 1_000_000.0)
    dock.auto_min_size_spin.setDecimals(1)
    dock.auto_min_size_spin.setValue(0.0)
    dock.auto_min_size_spin.setSuffix(" m²")
    dock.auto_min_size_spin.setSpecialValueText(tr("Off"))
    fit_spin_width(dock.auto_min_size_spin, 78, 110)
    dock.auto_min_size_spin.setToolTip(tr(
        "Hide detections smaller than this ground area. Use it to drop tiny "
        "noise blobs. 0 = keep all."))
    max_lbl = QLabel(tr("Maximum"))
    max_lbl.setObjectName("autoFieldLabel")
    dock.auto_max_size_spin = QDoubleSpinBox()
    dock.auto_max_size_spin.setRange(0.0, 10_000_000.0)
    dock.auto_max_size_spin.setDecimals(1)
    dock.auto_max_size_spin.setValue(0.0)
    dock.auto_max_size_spin.setSuffix(" m²")
    dock.auto_max_size_spin.setSpecialValueText(tr("No limit"))
    fit_spin_width(dock.auto_max_size_spin, 78, 110)
    dock.auto_max_size_spin.setToolTip(tr(
        "Hide detections larger than this ground area. 0 = no limit."))


    _box_w = max(dock.auto_min_size_spin.minimumWidth(),
                 dock.auto_max_size_spin.minimumWidth())
    for _spin in (dock.auto_min_size_spin, dock.auto_max_size_spin):
        _spin.setMinimumWidth(_box_w)



    for first, lbl, spin in ((True, min_lbl, dock.auto_min_size_spin),
                             (False, max_lbl, dock.auto_max_size_spin)):
        row, row_lay = review_list_row(first=first)
        row_lay.addWidget(lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        row_lay.addStretch(1)
        row_lay.addWidget(spin, 0, Qt.AlignmentFlag.AlignVCenter)
        col.addWidget(row)




    def _size_changed(changed: str) -> None:
        minimum, maximum = bounded_size_filter(
            dock.auto_min_size_spin.value(),
            dock.auto_max_size_spin.value(), changed)
        for widget, value in (
            (dock.auto_min_size_spin, minimum),
            (dock.auto_max_size_spin, maximum),
        ):
            if widget.value() == value:
                continue
            widget.blockSignals(True)
            try:
                widget.setValue(value)
            finally:
                widget.blockSignals(False)
        dock.auto_refine_changed.emit()

    dock.auto_min_size_spin.valueChanged.connect(
        lambda _v: _size_changed("minimum"))
    dock.auto_max_size_spin.valueChanged.connect(
        lambda _v: _size_changed("maximum"))
    dock.auto_keep_size_zone = block
    return block


def build_boundary_snap_block(dock) -> QWidget:











    from ...core.boundary_snap import snap_default_enabled

    dock._auto_boundary_snap_offered = False

    block, col = review_zone("autoKeepBordersZone")

    row = QHBoxLayout()
    row.setSpacing(8)
    row.addWidget(row_lead_glyph("join"), 0, Qt.AlignmentFlag.AlignVCenter)


    label = QLabel(tr("Shared borders:").rstrip(": \u00a0"))
    label.setObjectName("autoFieldLabel")
    tip = tr(
        "Give neighbouring shapes one exact border instead of a hairline "
        "gap or overlap. For land cover, where the map is one surface.")
    label.setToolTip(tip)
    dock.auto_boundary_snap_check = QCheckBox()
    dock.auto_boundary_snap_check.setChecked(snap_default_enabled())
    dock.auto_boundary_snap_check.setToolTip(tip)
    dock.auto_boundary_snap_check.stateChanged.connect(
        lambda s: dock._on_shape_control_changed(
            "shared_boundaries", bool(s)))
    row.addWidget(label)
    row.addStretch()
    row.addWidget(dock.auto_boundary_snap_check)
    col.addLayout(row)

    dock.auto_boundary_snap_hint = DismissibleHint(
        HINT_REVIEW_SHARED_BORDERS,
        tr("Closes the hairline gaps between neighbouring shapes, for "
           "land cover maps."),
        tint=BLUE_TINT,
    )

    dock.auto_boundary_snap_hint.set_flat(True)
    col.addWidget(dock.auto_boundary_snap_hint)




    dock.auto_boundary_snap_notice = QLabel("")
    dock.auto_boundary_snap_notice.setWordWrap(True)
    dock.auto_boundary_snap_notice.setObjectName("autoHint")
    dock.auto_boundary_snap_notice.setVisible(False)
    col.addWidget(dock.auto_boundary_snap_notice)

    dock.auto_boundary_snap_row = block
    block.setVisible(False)
    return block


def build_review_busy_row(dock) -> QWidget:








    block = QWidget()
    row = QHBoxLayout(block)


    row.setContentsMargins(8, 0, 8, 0)
    row.setSpacing(8)
    dock.auto_review_busy_label = QLabel(tr("Reshaping the objects..."))
    dock.auto_review_busy_label.setObjectName("autoHint")
    row.addWidget(dock.auto_review_busy_label)
    row.addStretch()
    dock.auto_review_stop_btn = QPushButton(tr("Stop"))
    dock.auto_review_stop_btn.setStyleSheet(_BTN_LINK_MUTED)
    dock.auto_review_stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    dock.auto_review_stop_btn.setToolTip(tr(
        "Stop reshaping here. The outlines already redrawn are kept."))
    row.addWidget(dock.auto_review_stop_btn)
    dock.auto_review_busy_row = block
    block.setVisible(False)
    return block


def set_review_busy(dock, busy: bool) -> None:





    row = getattr(dock, "auto_review_busy_row", None)
    if row is None:
        return
    try:
        row.setVisible(bool(busy))
    except (RuntimeError, AttributeError):
        pass


def set_boundary_snap_notice(dock, reason: str) -> None:





    label = getattr(dock, "auto_boundary_snap_notice", None)
    if label is None:
        return
    try:
        label.setText(reason or "")
        label.setVisible(bool(reason))
    except (RuntimeError, AttributeError):
        pass
