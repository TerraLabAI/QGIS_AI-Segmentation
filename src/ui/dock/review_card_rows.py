"""Blocks of the review card: the size filter, shared borders, and the
shape-pass line.

Built as free functions taking the dock, not as mixin methods, because
``auto_review_build.py`` had grown past its size band and these blocks are a
self-contained concern: each one builds its widgets, hangs the handles the
setters read off the dock, and returns the block.
"""
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
from .auto_correct_build import _REVIEW_HEADING_NOTE_QSS, _REVIEW_HEADING_QSS
from .font_scale import scale_px_length
from .guidance import BLUE_TINT, HINT_REVIEW_SHARED_BORDERS, DismissibleHint
from .styles import (
    _BTN_LINK_MUTED,
    _SUBCARD_MARGINS,
    _SUBCARD_QSS,
    _card_divider,
)


def review_zone(obj_name: str) -> tuple[QWidget, QVBoxLayout]:
    """One group of the review card as its own bordered box, and its column.

    The Keep step holds two filters that both apply, Confidence and size.
    Run together in the card body they read as one control with a stray pair
    of number boxes at the bottom, so each gets its own box, the way the two
    input cards on the setup step are told apart. ``_SUBCARD_QSS`` and not
    ``_CARD_QSS``: these sit INSIDE the review card, and a box with the
    parent's own fill and border reads as a seam, not as a box.
    """
    zone = QWidget()
    zone.setObjectName(obj_name)
    zone.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    zone.setStyleSheet(_SUBCARD_QSS.format(name=obj_name))
    col = QVBoxLayout(zone)
    col.setContentsMargins(*_SUBCARD_MARGINS)
    col.setSpacing(6)
    return zone, col


def build_size_filter_block(dock) -> QWidget:
    """Min / Max ground area: the second way an object is kept or dropped.

    Both hide detections client-side by true ground area (free, instant;
    0 = off / no limit), so they belong beside Confidence and not with the
    outline controls they used to sit under. The count line above names this
    filter when it, rather than Confidence, is what hides everything: the
    control it names now sits on the same step.

    The header says WHOSE size it is. "Size" alone read as the size of the
    run or of the zone; the two boxes measure one detected object at a time.
    """
    block, col = review_zone("autoKeepSizeZone")
    hdr = QHBoxLayout()
    size_lbl = QLabel(tr("Size of each object"))
    size_lbl.setStyleSheet(_REVIEW_HEADING_QSS)
    size_note = QLabel(tr("hide anything outside this range"))
    size_note.setWordWrap(True)
    size_note.setStyleSheet(_REVIEW_HEADING_NOTE_QSS)
    hdr.addWidget(size_lbl)
    hdr.addWidget(size_note, 1)
    col.addLayout(hdr)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(10)
    min_lbl = QLabel(tr("Minimum"))
    min_lbl.setStyleSheet("font-size: 11px;")
    dock.auto_min_size_spin = QDoubleSpinBox()
    dock.auto_min_size_spin.setRange(0.0, 1_000_000.0)
    dock.auto_min_size_spin.setDecimals(1)
    dock.auto_min_size_spin.setValue(0.0)
    dock.auto_min_size_spin.setSuffix(" m²")
    dock.auto_min_size_spin.setSpecialValueText(tr("Off"))
    dock.auto_min_size_spin.setMinimumWidth(78)
    dock.auto_min_size_spin.setMaximumWidth(scale_px_length(110))
    dock.auto_min_size_spin.setToolTip(tr(
        "Hide detections smaller than this ground area. Use it to drop tiny "
        "noise blobs. 0 = keep all."))
    max_lbl = QLabel(tr("Maximum"))
    max_lbl.setStyleSheet("font-size: 11px;")
    dock.auto_max_size_spin = QDoubleSpinBox()
    dock.auto_max_size_spin.setRange(0.0, 10_000_000.0)
    dock.auto_max_size_spin.setDecimals(1)
    dock.auto_max_size_spin.setValue(0.0)
    dock.auto_max_size_spin.setSuffix(" m²")
    dock.auto_max_size_spin.setSpecialValueText(tr("No limit"))
    dock.auto_max_size_spin.setMinimumWidth(78)
    dock.auto_max_size_spin.setMaximumWidth(scale_px_length(110))
    dock.auto_max_size_spin.setToolTip(tr(
        "Hide detections larger than this ground area. 0 = no limit."))
    row.addWidget(min_lbl)
    row.addWidget(dock.auto_min_size_spin)
    row.addStretch()
    row.addWidget(max_lbl)
    row.addWidget(dock.auto_max_size_spin)
    col.addLayout(row)

    # Re-derive the visible set on any change (confidence has its own
    # debounced re-filter path).
    dock.auto_min_size_spin.valueChanged.connect(
        lambda _v: dock.auto_refine_changed.emit())
    dock.auto_max_size_spin.valueChanged.connect(
        lambda _v: dock.auto_refine_changed.emit())
    dock.auto_keep_size_zone = block
    return block


def build_boundary_snap_block(dock) -> QWidget:
    """Shared borders: one checkbox plus its one-line tip.

    A land-cover map is one surface cut into classes, so neighbouring
    shapes are meant to meet exactly; a detector returns each shape on its
    own, leaving a hairline gap or overlap. This closes them, free and
    instantly, on the shapes currently shown.

    The whole block is HIDDEN unless the run looks like land cover (see
    set_boundary_snap_offered): between two buildings or two cars the gap
    is real data, so there the option must not exist at all.
    """
    from ...core.boundary_snap import snap_default_enabled

    dock._auto_boundary_snap_offered = False
    block = QWidget()
    col = QVBoxLayout(block)
    col.setContentsMargins(0, 0, 0, 0)
    col.setSpacing(4)
    col.addWidget(_card_divider())

    row = QHBoxLayout()
    label = QLabel(tr("Shared borders:"))
    label.setStyleSheet("font-size: 11px;")
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
    col.addWidget(dock.auto_boundary_snap_hint)

    # Why the pass did nothing, when it refuses a set (too many shapes, no
    # position to measure at). Without it the box stays ticked and the map
    # never changes, which reads as a broken control.
    dock.auto_boundary_snap_notice = QLabel("")
    dock.auto_boundary_snap_notice.setWordWrap(True)
    dock.auto_boundary_snap_notice.setStyleSheet(
        "font-size: 10px; color: rgba(128, 128, 128, 0.95);")
    dock.auto_boundary_snap_notice.setVisible(False)
    col.addWidget(dock.auto_boundary_snap_notice)

    dock.auto_boundary_snap_row = block
    block.setVisible(False)
    return block


def build_review_busy_row(dock) -> QWidget:
    """The line that says a shape pass is running, and the way to stop it.

    Moving a Shape control re-shapes every object of the run. On a dense
    result that takes a while, and nothing on screen said so: the map
    changed in waves with no explanation and no way out. This says what is
    happening and lets the user stop where it is, keeping the shapes
    already written.
    """
    block = QWidget()
    row = QHBoxLayout(block)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    dock.auto_review_busy_label = QLabel(tr("Reshaping the objects..."))
    dock.auto_review_busy_label.setStyleSheet(
        "font-size: 11px; color: rgba(128, 128, 128, 0.95);")
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
    """Show or hide the line saying a shape pass is running, with its Stop.

    A Shape control re-shapes every object of the run, and on a dense result
    the map changed in waves with nothing on screen to say why or how to stop
    it."""
    row = getattr(dock, "auto_review_busy_row", None)
    if row is None:
        return
    try:
        row.setVisible(bool(busy))
    except (RuntimeError, AttributeError):
        pass


def set_boundary_snap_notice(dock, reason: str) -> None:
    """Say why shared borders did nothing on this set, or clear the line.

    The pass can refuse a set (too many shapes, no position to measure at) and
    it used to do so only in the message log, leaving a ticked control with no
    effect on the map."""
    label = getattr(dock, "auto_boundary_snap_notice", None)
    if label is None:
        return
    try:
        label.setText(reason or "")
        label.setVisible(bool(reason))
    except (RuntimeError, AttributeError):
        pass
