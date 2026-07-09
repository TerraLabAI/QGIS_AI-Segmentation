





from __future__ import annotations

from contextlib import suppress

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
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
    REFINE_MIN_SIZE_M2_DEFAULT,
    REFINE_ORTHO_DEFAULT,
    REFINE_POINTS_PCT_DEFAULT,
    REFINE_SIMPLIFY_DEFAULT,
    REFINE_SMOOTH_DEFAULT,
    REFINE_SMOOTH_ITERATIONS,
)
from .fold_row import FoldRow
from .font_scale import fit_spin_width, scale_px_length, scale_qss_font_px
from .refine_persistence import (
    apply_refine_start_values,
    capture_refine_settings,
    refine_setting_name_for,
    refine_start_values,
    remember_refine_settings,
)
from .styles import (
    _CARD_MARGINS,
    _CARD_QSS,
    _FOLD_TITLE_QSS,
    _HINT_LINE_QSS,
    FONT_BODY,
    FONT_HINT,
    INK,
    INK_2,
    _card_divider,
    _settings_section,
    repaint_on_theme_change,
)







REFINE_SETTLE_DEFAULT_MS = 90
_MIN_REFINE_SETTLE_MS = 30
_MAX_REFINE_SETTLE_MS = 400


def refine_settle_ms() -> int:

    try:
        from ...core.server_dials import dial_in_range

        return int(dial_in_range(
            "ui.refine_settle_ms", REFINE_SETTLE_DEFAULT_MS,
            _MIN_REFINE_SETTLE_MS, _MAX_REFINE_SETTLE_MS))
    except Exception:  # noqa: BLE001  # nosec B110
        return REFINE_SETTLE_DEFAULT_MS





_REFINE_MORE_EXPANDED_KEY = "AISegmentation/refine/more_expanded"


def _refine_row_label(text: str, tooltip: str) -> QLabel:


    lbl = QLabel(text)
    lbl.setStyleSheet(scale_qss_font_px(
        f"font-size: {FONT_BODY}px; color: {INK};"))
    lbl.setToolTip(tooltip)
    return lbl




_ROW_MIN_PX = 32



_SPIN_W_PX = 110




_ROW_GLYPH_PX = 16
_ROW_GLYPH_GAP = 8


def _refine_control_row(icon_name: str, label: QLabel, control) -> QHBoxLayout:

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)


    row.setSpacing(_ROW_GLYPH_GAP)
    from ..icons import pixmap_for
    from .auto_flow_look import token_qcolor

    glyph = QLabel()

    def _paint_row_glyph(target, name=icon_name) -> None:

        target.setPixmap(pixmap_for(
            target, name, _ROW_GLYPH_PX, token_qcolor(INK)))

    _paint_row_glyph(glyph)
    repaint_on_theme_change(glyph, _paint_row_glyph)
    glyph.setFixedSize(_ROW_GLYPH_PX, _ROW_GLYPH_PX)

    glyph.setAccessibleName("")
    glyph.setStyleSheet("background: transparent; border: none;")
    row.addWidget(glyph, 0, Qt.AlignmentFlag.AlignVCenter)


    try:
        if not control.accessibleName():
            import re

            control.setAccessibleName(
                re.sub(r"<[^>]+>", "", label.text()).strip())
    except (AttributeError, RuntimeError, TypeError):
        pass  # nosec B110
    row.addWidget(label)
    row.addStretch()
    row.addWidget(control)
    return row


def _settings_row(layout: QHBoxLayout) -> QWidget:

    host = QWidget()
    host.setMinimumHeight(scale_px_length(_ROW_MIN_PX))
    host.setLayout(layout)
    return host


class DockRefineMixin:


    def _setup_refine_panel(self, parent_layout):











        self._refine_panel_title = tr("Outline settings")
        self.refine_group = QWidget()
        self.refine_group.setVisible(False)
        refine_layout = QVBoxLayout(self.refine_group)
        refine_layout.setSpacing(0)
        refine_layout.setContentsMargins(0, 0, 0, 0)


        self.refine_content_widget = QWidget()
        self.refine_content_widget.setObjectName("refineContentWidget")
        self.refine_content_widget.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)




        from .auto_flow_look import toggle_indicator_qss
        from .widgets import checkbox_indicator_qss, label_with_target_hint

        _refine_qss = _CARD_QSS.format(name="refineContentWidget")
        _refine_qss += "QLabel { background: transparent; border: none; }"
        _refine_qss += checkbox_indicator_qss(self)
        _refine_qss += toggle_indicator_qss(self)
        self.refine_content_widget.setStyleSheet(_refine_qss)
        refine_content_layout = QVBoxLayout(self.refine_content_widget)


        refine_content_layout.setContentsMargins(*_CARD_MARGINS)
        refine_content_layout.setSpacing(0)


        self.refine_title_label = QLabel(self._refine_panel_title)
        self.refine_title_label.setStyleSheet(_FOLD_TITLE_QSS)
        refine_content_layout.addWidget(self.refine_title_label)
        refine_content_layout.addSpacing(4)





        ortho_label = _refine_row_label(label_with_target_hint(
            tr("Right angles"), tr("buildings")), tr(
            "Snap walls to right angles, 45 degree walls included. Made for "
            "buildings, pools and solar panels. A shape it would distort is "
            "left as it is."))
        self.right_angles_checkbox = QCheckBox()
        self.right_angles_checkbox.setToolTip(ortho_label.toolTip())
        self.right_angles_checkbox.setChecked(REFINE_ORTHO_DEFAULT)


        self.right_angles_label = ortho_label
        ortho_row = _refine_control_row(
            "right_angle", ortho_label, self.right_angles_checkbox)

        round_label = _refine_row_label(label_with_target_hint(
            tr("Round corners"), tr("trees")), tr(
            "Round corners for natural shapes like trees and bushes. "
            "Lower Points for smoother results."))
        self.round_corners_checkbox = QCheckBox()
        self.round_corners_checkbox.setToolTip(round_label.toolTip())
        self.round_corners_checkbox.setChecked(REFINE_SMOOTH_DEFAULT > 0)
        round_row = _refine_control_row(
            "round_corner", round_label, self.round_corners_checkbox)

        fill_label = _refine_row_label(
            tr("Fill holes"), tr("Fill interior holes in the selection"))
        self.fill_holes_checkbox = QCheckBox()
        self.fill_holes_checkbox.setChecked(REFINE_FILL_HOLES_DEFAULT)
        self.fill_holes_checkbox.setToolTip(fill_label.toolTip())
        fill_row = _refine_control_row(
            "fill_holes", fill_label, self.fill_holes_checkbox)






        self.fill_holes_max_row = QWidget()
        fill_max_layout = QHBoxLayout(self.fill_holes_max_row)

        fill_max_layout.setContentsMargins(
            _ROW_GLYPH_PX + _ROW_GLYPH_GAP, 2, 0, 0)
        fill_max_layout.setSpacing(6)
        fill_max_label = _refine_row_label(tr("Up to"), tr(
            "Fill only holes smaller than this ground area. Bigger holes (a "
            "road median, a courtyard) stay open. No limit = fill every hole."))
        fill_max_label.setStyleSheet(scale_qss_font_px(
            f"font-size: {FONT_HINT}px; color: {INK_2};"))
        self.fill_holes_max_spinbox = QDoubleSpinBox()
        self.fill_holes_max_spinbox.setRange(0.0, 1_000_000.0)
        self.fill_holes_max_spinbox.setDecimals(1)
        self.fill_holes_max_spinbox.setValue(REFINE_FILL_HOLES_MAX_M2_DEFAULT)
        self.fill_holes_max_spinbox.setSuffix(" m²")
        self.fill_holes_max_spinbox.setSpecialValueText(tr("No limit"))
        self.fill_holes_max_spinbox.setToolTip(fill_max_label.toolTip())
        fit_spin_width(self.fill_holes_max_spinbox, _SPIN_W_PX, _SPIN_W_PX)
        fill_max_layout.addWidget(fill_max_label)
        fill_max_layout.addStretch()
        fill_max_layout.addWidget(self.fill_holes_max_spinbox)
        self.fill_holes_max_row.setVisible(REFINE_FILL_HOLES_DEFAULT)











        points_label = _refine_row_label(tr("Points"), tr(
            "Share of the outline's points to keep. 100% is the standard "
            "density.\nLower thins the smallest detail first, keeps the "
            "corners, and gives Right angles straight walls to square."))
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setSingleStep(5)
        self.points_spinbox.setRange(1, 100)
        self.points_spinbox.setValue(REFINE_POINTS_PCT_DEFAULT)
        self.points_spinbox.setSuffix(" %")
        fit_spin_width(self.points_spinbox, _SPIN_W_PX, _SPIN_W_PX)
        self.points_spinbox.setToolTip(points_label.toolTip())
        points_row = _refine_control_row(
            "points", points_label, self.points_spinbox)




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
        fit_spin_width(self.simplify_spinbox, _SPIN_W_PX, _SPIN_W_PX)
        self.simplify_spinbox.setToolTip(simplify_label.toolTip())
        simplify_row = _refine_control_row(
            "simplify", simplify_label, self.simplify_spinbox)




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
        fit_spin_width(self.clean_edges_spinbox, _SPIN_W_PX, _SPIN_W_PX)
        clean_row = _refine_control_row(
            "trim", clean_label, self.clean_edges_spinbox)

        expand_label = _refine_row_label(tr("Grow / shrink"), tr(
            "Positive = grow outward, negative = shrink inward"))
        self.expand_spinbox = QSpinBox()
        self.expand_spinbox.setRange(-1000, 1000)
        self.expand_spinbox.setValue(REFINE_EXPAND_DEFAULT)
        self.expand_spinbox.setSuffix(" px")
        self.expand_spinbox.setToolTip(expand_label.toolTip())
        fit_spin_width(self.expand_spinbox, _SPIN_W_PX, _SPIN_W_PX)
        expand_row = _refine_control_row(
            "grow_shrink", expand_label, self.expand_spinbox)





        self._refine_right_angle_conflicts = (
            clean_label,
            self.clean_edges_spinbox,
            round_label,
            self.round_corners_checkbox,
        )
        self._refine_right_angle_conflict_tooltips = tuple(
            (widget, widget.toolTip())
            for widget in self._refine_right_angle_conflicts)








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
        fit_spin_width(self.min_size_spinbox, _SPIN_W_PX, _SPIN_W_PX)




        size_row = _settings_row(_refine_control_row(
            "size_range", min_size_label, self.min_size_spinbox))





        self.refine_shape_toggles = QWidget()
        top_rows = QVBoxLayout(self.refine_shape_toggles)
        top_rows.setContentsMargins(0, 0, 0, 0)
        top_rows.setSpacing(0)
        top_rows.addWidget(_settings_row(ortho_row))
        top_rows.addWidget(_card_divider())
        top_rows.addWidget(_settings_row(round_row))
        top_rows.addWidget(_card_divider())
        refine_content_layout.addWidget(self.refine_shape_toggles)
        refine_content_layout.addWidget(_settings_row(fill_row))
        refine_content_layout.addWidget(self.fill_holes_max_row)
        refine_content_layout.addSpacing(2)
        refine_content_layout.addWidget(_card_divider())



        self.refine_more_btn = FoldRow(
            tr("More settings"), settings_key=_REFINE_MORE_EXPANDED_KEY)
        self.refine_more_btn.set_fold_fact(
            tr("Points, Simplify, Trim, Grow, Size"))
        refine_content_layout.addSpacing(2)
        refine_content_layout.addWidget(self.refine_more_btn)

        self.refine_more_widget = QWidget()
        self.refine_more_widget.setObjectName("refineMoreWidget")
        self.refine_more_widget.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        self.refine_more_widget.setStyleSheet(
            "QWidget#refineMoreWidget { background: transparent; border: none; }"
            "QLabel { background: transparent; border: none; }")
        more_layout = QVBoxLayout(self.refine_more_widget)
        more_layout.setContentsMargins(0, 2, 0, 2)
        more_layout.setSpacing(0)



        self.refine_more_cloud_note = QLabel(tr(
            "Reshapes the outline only: no new AI run, no credits."))
        self.refine_more_cloud_note.setWordWrap(True)
        self.refine_more_cloud_note.setStyleSheet(_HINT_LINE_QSS)
        more_layout.addWidget(self.refine_more_cloud_note)
        more_layout.addSpacing(4)
        for spin_row in (points_row, simplify_row, clean_row, expand_row):
            more_layout.addWidget(_settings_row(spin_row))
        more_layout.addSpacing(6)


        more_layout.addWidget(_settings_section(tr("Size"), "", [size_row]))



        refine_content_layout.addWidget(self.refine_more_widget)
        self.refine_more_btn.bind_fold_body(self.refine_more_widget)
        self.refine_more_btn.fold_toggled.connect(
            self._on_refine_more_toggled)

        refine_layout.addWidget(self.refine_content_widget)




        for widget in (self.refine_group, self.refine_content_widget):
            widget.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)


        self.points_spinbox.valueChanged.connect(self._on_refine_changed)
        self.simplify_spinbox.valueChanged.connect(self._on_refine_changed)
        self.clean_edges_spinbox.valueChanged.connect(self._on_refine_changed)
        self.round_corners_checkbox.stateChanged.connect(self._on_refine_ticked)




        self.right_angles_checkbox.stateChanged.connect(
            self._sync_refine_right_angle_controls)
        self.right_angles_checkbox.stateChanged.connect(self._on_refine_ticked)
        self.expand_spinbox.valueChanged.connect(self._on_refine_changed)
        self.fill_holes_checkbox.stateChanged.connect(
            self._sync_fill_holes_max_row)
        self.fill_holes_checkbox.stateChanged.connect(self._on_refine_ticked)
        self.fill_holes_max_spinbox.valueChanged.connect(self._on_refine_changed)
        self.min_size_spinbox.valueChanged.connect(self._on_refine_changed)

        self._sync_refine_right_angle_controls()




        self._apply_refine_start_values()
        self._sync_refine_shape_toggles()
        parent_layout.addWidget(self.refine_group)

    def _on_refine_more_toggled(self, _open: bool) -> None:







        self._refresh_refine_fold_geometry()
        safe_single_shot(0, self, self._refresh_refine_fold_geometry)

    def _refresh_refine_fold_geometry(self) -> None:
        for widget in (getattr(self, "refine_more_widget", None),
                       getattr(self, "refine_content_widget", None),
                       getattr(self, "refine_group", None)):
            if widget is not None:
                try:
                    widget.updateGeometry()
                    layout = widget.layout()
                    if layout is not None:
                        layout.activate()
                except RuntimeError:
                    pass



    def _refine_start_values(self) -> dict:






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

        start = self._refine_start_values()
        apply_refine_start_values(self, start)
        self._forget_round_corners_memory()
        self._sync_fill_holes_max_row()
        if start.get("right_angles"):





            safe_single_shot(0, self, self._sync_refine_right_angle_controls)
        else:
            self._sync_refine_right_angle_controls()

    def publish_refine_settings(self) -> None:












        if getattr(self, "_publishing_refine_settings", False):
            return
        if getattr(self, "_refine_handoff", False):
            return
        self._publishing_refine_settings = True
        try:




            timer = getattr(self, "_refine_debounce_timer", None)
            if timer is not None:
                try:
                    timer.stop()
                except (RuntimeError, AttributeError):
                    pass
            self._emit_refine_changed()
        except (RuntimeError, AttributeError):
            pass  # nosec B110
        finally:
            self._publishing_refine_settings = False

    def _mark_refine_touched(self) -> None:








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






        if getattr(self, "_refine_handoff", False):
            return
        touched = getattr(self, "_refine_touched_keys", None)
        if not touched:
            return
        settled = capture_refine_settings(self, only=touched)




        held_round = getattr(self, "_round_corners_before_right_angles", None)
        if held_round is not None and "round_corners" in settled:
            settled["round_corners"] = bool(held_round)



        for key, memory_name in (
                ("right_angles", "_right_angles_before_offline"),
                ("round_corners", "_round_corners_before_offline")):
            held = getattr(self, memory_name, None)
            if held is not None and key in settled:
                settled[key] = bool(held)
        remember_refine_settings(settled)

    def _sync_fill_holes_max_row(self, _state=None) -> None:


        row = getattr(self, "fill_holes_max_row", None)
        if row is not None:
            row.setVisible(self.fill_holes_checkbox.isChecked())

    def _refine_shape_toggles_allowed(self) -> bool:











        if getattr(self, "_refine_handoff", False):
            return True
        try:
            from ...core.manual_cloud_route import manual_cloud_route_offered

            if not manual_cloud_route_offered():
                return True
            return bool(self._manual_cloud_route_picked())
        except (RuntimeError, AttributeError, ImportError):
            return True

    def _refine_cloud_note_wanted(self) -> bool:







        if getattr(self, "_refine_handoff", False):
            return False
        try:
            return bool(self._manual_cloud_route_picked())
        except (RuntimeError, AttributeError):
            return False

    def _sync_refine_shape_toggles(self, publish: bool = True) -> None:













        rows = getattr(self, "refine_shape_toggles", None)
        if rows is None:
            return
        allowed = self._refine_shape_toggles_allowed()
        rows.setVisible(allowed)
        note = getattr(self, "refine_more_cloud_note", None)
        if note is not None:
            try:
                note.setVisible(self._refine_cloud_note_wanted())
            except RuntimeError:
                pass
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



        if changed and publish:
            self.publish_refine_settings()

    def _sync_refine_right_angle_controls(self, _state=None) -> None:







        from .right_angles_support import apply_right_angle_conflicts

        enabled = apply_right_angle_conflicts(
            self, "right_angles_checkbox", "right_angles_label",
            "_refine_right_angle_conflict_tooltips")






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






        self._right_angles_before_offline = None
        self._round_corners_before_offline = None

    def _forget_round_corners_memory(self) -> None:






        self._round_corners_before_right_angles = None

    def _on_refine_changed(self, value=None):







        self._mark_refine_touched()
        self._refine_debounce_timer.start(refine_settle_ms())

    def _on_refine_ticked(self, _state=None):






        self._mark_refine_touched()
        self._refine_debounce_timer.stop()
        self._emit_refine_changed()

    def _emit_refine_changed(self):









        shape_allowed = self._refine_shape_toggles_allowed()
        right_angles = shape_allowed and self.right_angles_checkbox.isChecked()



        self._remember_refine_settings()
        self.size_filter_changed.emit(
            float(self.min_size_spinbox.value()),
            0.0,
        )
        self.fill_holes_size_changed.emit(
            float(self.fill_holes_max_spinbox.value()))



        self.clean_edges_changed.emit(
            0.0 if right_angles else float(self.clean_edges_spinbox.value()))
        self.outline_budget_changed.emit(
            float(self.simplify_spinbox.value()),
            int(self.points_spinbox.value()),
        )


        self.refine_settings_changed.emit(
            int(round(self.simplify_spinbox.value())),
            0 if right_angles or not shape_allowed else (
                REFINE_SMOOTH_ITERATIONS
                if self.round_corners_checkbox.isChecked() else 0),
            self.expand_spinbox.value(),
            self.fill_holes_checkbox.isChecked(),
            right_angles,
        )

    def reset_refine_sliders(self):








        start = self._refine_start_values()
        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox,
                  self.min_size_spinbox):
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

        for w in (self.points_spinbox, self.simplify_spinbox,
                  self.clean_edges_spinbox,
                  self.round_corners_checkbox,
                  self.right_angles_checkbox,
                  self.expand_spinbox, self.fill_holes_checkbox,
                  self.fill_holes_max_spinbox,
                  self.min_size_spinbox):
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











        del min_area
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
        self._sync_refine_shape_toggles(publish=False)

    def set_size_filter_values(self, min_m2: float, max_m2: float) -> None:





        del max_m2
        self.min_size_spinbox.blockSignals(True)
        self.min_size_spinbox.setValue(max(0.0, float(min_m2 or 0.0)))
        self.min_size_spinbox.blockSignals(False)
