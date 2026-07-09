"""Refine panel: build, expand/collapse, slider values and resets.

Part of AISegmentationDockWidget (see ai_segmentation_dockwidget.py);
split out so agents and humans work on one concern per file. Methods
are plain mixin members: widgets/signals live on the dock instance.
"""
from __future__ import annotations


from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


from ...core.i18n import tr
from ...core.qt_compat import event_pos
from .styles import (
    _REFINE_COLLAPSED_HEIGHT,
)


class DockRefineMixin:
    """Refine panel: build, expand/collapse, slider values and resets."""

    def _setup_refine_panel(self, parent_layout):
        """Setup the collapsible Refine mask panel (collapsible via click on title)."""
        self.refine_group = QGroupBox("▼ " + tr("Refine selection"))
        self.refine_group.setCheckable(False)  # No checkbox, just clickable title
        self.refine_group.setVisible(False)  # Hidden until segmentation active
        self.refine_group.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refine_group.mousePressEvent = self._on_refine_group_clicked
        # Remove all QGroupBox styling - make it look like a simple collapsible section
        self.refine_group.setStyleSheet("""
            QGroupBox {
                background-color: transparent;
                border: none;
                border-radius: 0px;
                margin: 0px;
                padding: 0px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                padding: 2px 4px;
                background-color: transparent;
                border: none;
            }
        """)
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(0)
        refine_layout.setContentsMargins(0, 0, 0, 0)

        # Content widget to show/hide - styled as a subtle bordered box
        self.refine_content_widget = QWidget()
        self.refine_content_widget.setObjectName("refineContentWidget")

        # Shared visible checkbox indicators (off = outlined box, on = blue
        # check): the old inline painter drew the unchecked state as a plain
        # background square, i.e. invisible.
        from .widgets import checkbox_indicator_qss
        _cb_qss = checkbox_indicator_qss(self)

        self.refine_content_widget.setStyleSheet(f"""
            QWidget#refineContentWidget {{
                background-color: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 128, 128, 0.2);
                border-radius: 4px;
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
            {_cb_qss}
        """)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)
        refine_content_layout.setContentsMargins(10, 10, 10, 10)
        refine_content_layout.setSpacing(6)

        # ── Outline section ──
        outline_label = QLabel(tr("Outline").upper())
        outline_label.setStyleSheet(
            "font-size: 10px; color: palette(text); font-weight: bold; "
            "background: transparent; border: none; border-bottom: 1px solid rgba(128, 128, 128, 0.35); "
            "padding: 8px 0px 4px 0px; margin-bottom: 4px; letter-spacing: 1px;")
        refine_content_layout.addWidget(outline_label)

        # 1. Simplify outline: SpinBox (0 to 1000) - reduces small variations
        simplify_layout = QHBoxLayout()
        simplify_label = QLabel(tr("Simplify outline:"))
        simplify_label.setToolTip(tr("Reduce small variations in the outline (0 = no change)"))
        self.simplify_spinbox = QSpinBox()
        self.simplify_spinbox.setRange(0, 1000)
        self.simplify_spinbox.setValue(3)
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
        self.round_corners_checkbox.setChecked(False)
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
        self.right_angles_checkbox.setChecked(False)
        ortho_layout.addWidget(ortho_label)
        ortho_layout.addStretch()
        ortho_layout.addWidget(self.right_angles_checkbox)
        refine_content_layout.addLayout(ortho_layout)

        # ── Selection section ──
        selection_label = QLabel(tr("Selection").upper())
        selection_label.setStyleSheet(
            "font-size: 10px; color: palette(text); font-weight: bold; "
            "background: transparent; border: none; border-bottom: 1px solid rgba(128, 128, 128, 0.35); "
            "padding: 14px 0px 4px 0px; margin-bottom: 4px; letter-spacing: 1px;")
        refine_content_layout.addWidget(selection_label)

        # 3. Expand/Contract: SpinBox with +/- buttons (-1000 to +1000)
        expand_layout = QHBoxLayout()
        expand_label = QLabel(tr("Expand/Contract:"))
        expand_label.setToolTip(tr("Positive = expand outward, Negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(0)
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
        self.fill_holes_checkbox.setChecked(True)
        self.fill_holes_checkbox.setToolTip(fill_label.toolTip())
        fill_layout.addWidget(fill_label)
        fill_layout.addStretch()
        fill_layout.addWidget(self.fill_holes_checkbox)
        refine_content_layout.addLayout(fill_layout)

        # Min area was previously exposed here but is now auto-computed based
        # on the crop scale (see plugin._compute_auto_min_area). Removing the
        # spinbox keeps the refine panel focused on the controls users tune.

        refine_layout.addWidget(self.refine_content_widget)
        self.refine_content_widget.setVisible(self._refine_expanded)
        # Set initial max height constraint (collapsed by default)
        if not self._refine_expanded:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)

        # Connect signals
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.round_corners_checkbox.stateChanged.connect(self._on_refine_changed)
        self.right_angles_checkbox.stateChanged.connect(self._on_refine_changed)
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_changed)

        parent_layout.addWidget(self.refine_group)

    def _on_refine_group_clicked(self, event):
        """Toggle the refine panel expanded/collapsed state (only when clicking on title)."""
        # Only toggle if click is in the title area
        # This prevents collapsing when clicking spinbox arrows at min/max values
        pt = event_pos(event)
        if pt.y() > _REFINE_COLLAPSED_HEIGHT:
            return  # Click was on content, not title - ignore

        self._refine_expanded = not self._refine_expanded
        arrow = "▼" if self._refine_expanded else "▶"
        self.refine_group.setTitle(f"{arrow} " + tr("Refine selection"))
        # Defer visibility and height changes to next event loop tick to avoid layout glitch
        expanded = self._refine_expanded
        QTimer.singleShot(0, lambda: self._apply_refine_toggle(expanded))

    def _apply_refine_toggle(self, expanded):
        """Apply refine panel expand/collapse after event loop tick."""
        self.refine_content_widget.setVisible(expanded)
        if expanded:
            self.refine_group.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self.refine_group.setMaximumHeight(_REFINE_COLLAPSED_HEIGHT)

    def _on_refine_changed(self, value=None):
        """Handle refine control changes with debounce."""
        self._refine_debounce_timer.start(150)

    def _emit_refine_changed(self):
        """Emit the refine settings changed signal after debounce."""
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
                  self.expand_spinbox, self.fill_holes_checkbox):
            w.blockSignals(True)

        self.simplify_spinbox.setValue(3)
        self.round_corners_checkbox.setChecked(False)
        self.right_angles_checkbox.setChecked(False)
        self.expand_spinbox.setValue(0)
        self.fill_holes_checkbox.setChecked(True)

        for w in (self.simplify_spinbox, self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox):
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
