










from __future__ import annotations

from qgis.PyQt.QtCore import QRectF, QSize, Qt, QTimer
from qgis.PyQt.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import (
    QAbstractButton,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core.i18n import tr
from .auto_flow_look import _BTN_AUTO_QUIET
from .font_scale import scale_px_length, scale_qss_font_px
from .session_link_row import SessionLinkRow
from .styles import (
    _BTN_GREEN_STEP,
    _CARD_MARGINS,
    BTN_PRIMARY_WIDE_PX,
    FONT_BASE,
    FONT_BODY,
    FONT_HINT,
    HOVER,
    INK,
    INK_2,
    INK_3,
    LINE,
    SPACE_CARD,
    SURFACE,
    spin_theme_qss,
)
from .ui_refresh_credits import grouped_locale

__all__ = ["DockLandCoverCardMixin", "land_cover_area_text", "land_cover_share_text"]


_PATCH_DEBOUNCE_MS = 450
_SWATCH_PX = 14
_BAR_PX = 8
_BAR_GAP_PX = 2



_MIN_AREA_PRESETS = (0, 20, 50, 200, 500)


def land_cover_area_text(m2: float) -> str:


    m2 = max(0.0, float(m2))
    loc = grouped_locale()
    if m2 >= 10000.0:
        return tr("{n} ha").format(n=loc.toString(m2 / 10000.0, "f", 2))
    return tr("{n} m²").format(n=loc.toString(int(round(m2))))


def land_cover_share_text(pct: int) -> str:

    return tr("{n}%").format(n=int(pct))


def _qcolor(hex_value: str, alpha: int = 255) -> QColor:
    color = QColor(hex_value)
    if not color.isValid():
        color = QColor(128, 128, 128)
    color.setAlpha(alpha)
    return color


class _LandCoverBar(QWidget):


    def __init__(self, parent=None):
        super().__init__(parent)
        self._parts: list[tuple[str, float]] = []
        self.setFixedHeight(scale_px_length(_BAR_PX))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_parts(self, parts: list[tuple[str, float]]) -> None:
        self._parts = [(c, float(v)) for c, v in parts if v > 0]
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = QRectF(self.rect())
            radius = rect.height() / 2.0
            clip = QPainterPath()
            clip.addRoundedRect(rect, radius, radius)
            painter.setClipPath(clip)
            painter.setPen(Qt.PenStyle.NoPen)
            total = sum(v for _c, v in self._parts)
            if total <= 0:
                painter.setBrush(QColor(128, 128, 128, 50))
                painter.drawRect(rect)
                painter.end()
                return
            gap = float(scale_px_length(_BAR_GAP_PX))
            room = rect.width() - gap * max(0, len(self._parts) - 1)
            x = rect.left()
            for i, (color, value) in enumerate(self._parts):
                w = room * value / total
                if i == len(self._parts) - 1:
                    w = rect.right() - x
                painter.setBrush(_qcolor(color))
                painter.drawRect(QRectF(x, rect.top(), max(w, 0.0), rect.height()))
                if w >= 2:
                    from .auto_target_mode import land_cover_edge_color
                    painter.setPen(QPen(land_cover_edge_color(), 1))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRectF(x + 0.5, rect.top() + 0.5,
                                            max(w - 1.0, 0.0), rect.height() - 1.0))
                    painter.setPen(Qt.PenStyle.NoPen)
                x += w + gap
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


class _PresetSpin(QDoubleSpinBox):



    def stepBy(self, steps):  # noqa: N802
        value = self.value()
        if steps > 0:
            ahead = [p for p in _MIN_AREA_PRESETS if p > value + 1e-9]
            target = ahead[min(steps, len(ahead)) - 1] if ahead else _MIN_AREA_PRESETS[-1]
        else:
            behind = [p for p in _MIN_AREA_PRESETS if p < value - 1e-9]
            target = behind[-min(-steps, len(behind))] if behind else _MIN_AREA_PRESETS[0]
        self.setValue(float(target))

    def stepEnabled(self):  # noqa: N802
        flags = QDoubleSpinBox.StepEnabledFlag
        out = QDoubleSpinBox.StepEnabledFlag.StepNone
        if self.value() > _MIN_AREA_PRESETS[0]:
            out |= flags.StepDownEnabled
        if self.value() < _MIN_AREA_PRESETS[-1]:
            out |= flags.StepUpEnabled
        return out


class _EyeButton(QAbstractButton):



    def __init__(self, parent=None):
        super().__init__(parent)
        self._shown = True
        side = scale_px_length(24)
        self.setFixedSize(side, side)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setToolTip(tr("Show/hide on map and in export"))
        self.setAccessibleName(tr("Show/hide on map and in export"))

    def set_shown(self, shown: bool) -> None:
        self._shown = bool(shown)
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            if self.underMouse() or self.hasFocus():
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(128, 128, 128, 40))
                painter.drawRoundedRect(outer, 6, 6)
            if self.hasFocus():
                painter.setPen(QPen(QColor("#1e88e5"), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(outer, 6, 6)
            ink = QColor(INK if (self._shown or self.underMouse()) else INK_3)
            pen = QPen(ink, 1.4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            c = outer.center()
            w = outer.width() * 0.62
            h = outer.height() * 0.36
            path = QPainterPath()
            path.moveTo(c.x() - w / 2, c.y())
            path.quadTo(c.x(), c.y() - h, c.x() + w / 2, c.y())
            path.quadTo(c.x(), c.y() + h, c.x() - w / 2, c.y())
            painter.drawPath(path)
            r = outer.width() * 0.11
            if self._shown:
                painter.setBrush(ink)
            painter.drawEllipse(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
            if not self._shown:
                painter.drawLine(QRectF(outer).adjusted(6, 6, -6, -6).bottomLeft(),
                                 QRectF(outer).adjusted(6, 6, -6, -6).topRight())
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


class _ClassSwatch(QAbstractButton):



    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = "#808080"
        self._shown = True
        side = scale_px_length(_SWATCH_PX + 4)
        self.setFixedSize(side, side)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def sizeHint(self):  # noqa: N802
        side = scale_px_length(_SWATCH_PX + 4)
        return QSize(side, side)

    def set_state(self, color: str, shown: bool, empty: bool = False) -> None:
        self._color = color
        self._shown = bool(shown)
        self._empty = bool(empty)
        self.update()

    def paintEvent(self, _event):  # noqa: N802
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            if self.underMouse() or self.hasFocus():
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(128, 128, 128, 40))
                painter.drawRoundedRect(outer, 6, 6)
            if self.hasFocus():
                painter.setPen(QPen(QColor("#1e88e5"), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(outer, 6, 6)
            side = float(scale_px_length(_SWATCH_PX))
            inner = QRectF(0, 0, side, side)
            inner.moveCenter(outer.center())

            color = _qcolor(self._color, 110 if getattr(self, "_empty", False) else 255)
            if self._shown:
                from .auto_target_mode import land_cover_edge_color
                painter.setPen(QPen(land_cover_edge_color(), 1))
                painter.setBrush(color)
            else:
                painter.setPen(QPen(color, 1.6))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                inner = inner.adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(inner, 4, 4)
            painter.end()
        except Exception:  # noqa: BLE001  # nosec B110
            pass


class _ClassRow(QWidget):


    def __init__(self, on_hover, on_toggle, parent=None):
        super().__init__(parent)
        self.class_id = -1
        self._on_hover = on_hover
        self.setObjectName("lcClassRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 6, 1)
        lay.setSpacing(6)
        self.eye = _EyeButton(self)
        self.eye.clicked.connect(lambda: on_toggle(self.class_id))
        self.swatch = _ClassSwatch(self)
        self.name = QLabel(self)
        self.name.setObjectName("lcName")
        self.name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.area = QLabel(self)
        self.area.setObjectName("lcArea")
        self.area.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.share = QLabel(self)
        self.share.setObjectName("lcShare")
        self.share.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.eye)
        lay.addWidget(self.swatch)
        lay.addWidget(self.name, 1)
        lay.addWidget(self.area)
        lay.addWidget(self.share)

    def enterEvent(self, event):  # noqa: N802
        try:
            self._on_hover(self.class_id)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        try:
            self._on_hover(None)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
        super().leaveEvent(event)


def _land_cover_card_qss() -> str:
    return scale_qss_font_px(
        f"QWidget#autoLandCoverCard {{ background-color: {SURFACE};"
        f" border: 1px solid {LINE}; border-radius: 12px; }}"
        "QWidget#autoLandCoverCard QLabel { background: transparent; border: none; }"
        f"QLabel#lcTitle {{ color: {INK}; font-size: {FONT_BASE}px; font-weight: 600; }}"
        f"QLabel#lcTotal {{ color: {INK_2}; font-size: {FONT_BODY}px; }}"
        f"QLabel#lcSummary {{ color: {INK_2}; font-size: {FONT_BODY}px; }}"
        f"QWidget#lcClassRow {{ background: transparent; border-radius: 8px; }}"
        f"QWidget#lcClassRow:hover {{ background-color: {HOVER}; }}"
        f"QLabel#lcName {{ color: {INK}; font-size: {FONT_BODY}px; font-weight: 500; }}"
        f'QLabel#lcName[empty="true"] {{ color: {INK_3}; font-weight: 400; }}'
        f"QLabel#lcArea {{ color: {INK}; font-size: {FONT_BODY}px; }}"
        f'QLabel#lcArea[empty="true"] {{ color: {INK_3}; }}'
        f"QLabel#lcShare {{ color: {INK_2}; font-size: {FONT_HINT}px; }}"
        f'QLabel#lcShare[empty="true"] {{ color: {INK_3}; }}'
        f"QWidget#lcDivider {{ background-color: {LINE}; }}"
        f"QLabel#lcPatchLabel {{ color: {INK}; font-size: {FONT_BODY}px; }}"
        f"QLabel#lcName:disabled, QLabel#lcArea:disabled, QLabel#lcShare:disabled"
        f" {{ color: {INK_3}; }}"
    )


class DockLandCoverCardMixin:


    def _build_land_cover_card(self) -> None:

        if self.__dict__.get("auto_land_cover_panel") is not None:
            return
        panel = QWidget()
        panel.setObjectName("autoLandCoverPanel")
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)

        card = QWidget()
        card.setObjectName("autoLandCoverCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(_land_cover_card_qss())
        lay = QVBoxLayout(card)
        lay.setContentsMargins(*_CARD_MARGINS)
        lay.setSpacing(SPACE_CARD + 2)

        head = QHBoxLayout()
        head.setSpacing(8)
        title = QLabel(tr("Land cover"))
        title.setObjectName("lcTitle")
        self._lc_total_label = QLabel("")
        self._lc_total_label.setObjectName("lcTotal")
        self._lc_total_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lc_total_label.setToolTip(tr("Area of the zone"))
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(self._lc_total_label)
        lay.addLayout(head)

        self._lc_bar = _LandCoverBar()
        lay.addWidget(self._lc_bar)


        self._lc_summary = QLabel("")
        self._lc_summary.setObjectName("lcSummary")
        self._lc_summary.setWordWrap(True)
        lay.addWidget(self._lc_summary)




        self._lc_rows_host = QWidget()
        self._lc_rows_box = QVBoxLayout(self._lc_rows_host)
        self._lc_rows_box.setSpacing(0)
        self._lc_rows_box.setContentsMargins(0, 2, 0, 2)
        lay.addWidget(self._lc_rows_host)
        self._lc_rows: list[_ClassRow] = []

        divider = QWidget()
        divider.setObjectName("lcDivider")
        divider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        divider.setFixedHeight(1)
        lay.addWidget(divider)

        patch = QHBoxLayout()
        patch.setSpacing(8)
        patch_label = QLabel(tr("Minimum area"))
        patch_label.setObjectName("lcPatchLabel")
        patch_label.setToolTip(tr(
            "Smaller patches merge into their neighbour (the minimum "
            "mapping unit). No hole is left."))
        patch_label.setWordWrap(True)
        patch_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._lc_patch_spin = _PresetSpin()
        self._lc_patch_spin.setDecimals(0)
        self._lc_patch_spin.setRange(0.0, 100000.0)
        self._lc_patch_spin.setSuffix(" " + tr("m²"))
        self._lc_patch_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lc_patch_spin.setKeyboardTracking(False)
        self._lc_patch_spin.setStyleSheet(scale_qss_font_px(spin_theme_qss()))
        self._lc_patch_spin.setMinimumWidth(scale_px_length(84))
        self._lc_patch_spin.setToolTip(patch_label.toolTip())
        self._lc_patch_timer = QTimer(card)
        self._lc_patch_timer.setSingleShot(True)
        self._lc_patch_timer.setInterval(_PATCH_DEBOUNCE_MS)
        self._lc_patch_timer.timeout.connect(self._on_lc_patch_settled)
        self._lc_patch_spin.valueChanged.connect(
            lambda _v: self._lc_patch_timer.start())
        patch.addWidget(patch_label, 1)
        patch.addWidget(self._lc_patch_spin)
        lay.addLayout(patch)
        col.addWidget(card)

        self._lc_export_btn = QPushButton(tr("Export land cover"))
        self._lc_export_btn.setStyleSheet(_BTN_GREEN_STEP)
        self._lc_export_btn.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
        self._lc_export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lc_export_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._lc_export_btn.setAutoDefault(False)
        self._lc_export_btn.clicked.connect(lambda: self._lc_action("export"))
        col.addWidget(self._lc_export_btn)

        links = SessionLinkRow()
        for key, text, tip in (
                ("copy", tr("Copy table"),
                 tr("Copy the classes, areas and shares as a table you can "
                    "paste into a spreadsheet.")),
                ("rerun", tr("Re-run the whole zone"),
                 tr("Go back to your zone and run it again. Nothing is saved.")),
                ("exit", tr("Exit"), "")):
            btn = QPushButton(text)
            btn.setStyleSheet(_BTN_AUTO_QUIET)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            btn.setAutoDefault(False)
            if tip:
                btn.setToolTip(tip)
            btn.clicked.connect(lambda _c=False, k=key: self._lc_action(k))


            btn.ensurePolished()
            font = btn.font()
            font.setWeight(font.Weight.DemiBold)
            btn.setMinimumWidth(
                QFontMetrics(font).horizontalAdvance(text) + scale_px_length(20))
            links.add_session_link(btn)
        col.addWidget(links)

        panel.setVisible(False)
        self.auto_land_cover_panel = panel
        self._auto_review_column_layout.insertWidget(0, panel)

    def _lc_action(self, key: str, *args) -> None:
        actions = self.__dict__.get("_lc_actions") or {}
        fn = actions.get(key)
        if fn is not None:
            fn(*args)

    def _on_lc_patch_settled(self) -> None:
        self._lc_action("min_patch", float(self._lc_patch_spin.value()))

    def _land_cover_review_parts(self) -> list:

        names = ("_auto_review_card", "auto_step_next_btn", "auto_export_btn",
                 "auto_retry_btn", "_auto_review_links_sep",
                 "auto_review_exit_btn", "auto_review_free_fit_card",
                 "auto_review_view_row")
        return [w for w in (self.__dict__.get(n) for n in names) if w is not None]

    def show_land_cover_result(self, active: bool) -> None:


        if active:


            self._build_land_cover_card()
            self.set_auto_review_active(True)
            for widget in self._land_cover_review_parts():
                widget.setVisible(False)
            self.set_land_cover_error(False)
            self.auto_land_cover_panel.setVisible(True)
            return
        panel = self.__dict__.get("auto_land_cover_panel")
        if panel is not None:
            panel.setVisible(False)
            self._lc_patch_timer.stop()
        box = self.__dict__.get("_lc_error_box")
        if box is not None:
            box.setVisible(False)

    def set_land_cover_error(self, on: bool) -> None:



        box = self.__dict__.get("_lc_error_box")
        if not on:
            if box is not None:
                box.setVisible(False)
            return
        if box is None:
            from .styles import msg_rich
            box = QWidget()
            box.setObjectName("autoLandCoverError")
            col = QVBoxLayout(box)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(8)
            from html import escape
            text = QLabel(msg_rich("error", (
                "<b>" + escape(tr("Couldn't build the land cover map")) + "</b><br>"
                + escape(tr("Your run is kept. Try again: nothing is sent or charged."))),
                is_html=True))
            text.setWordWrap(True)
            col.addWidget(text)
            retry = QPushButton(tr("Try again"))
            retry.setStyleSheet(_BTN_GREEN_STEP)
            retry.setMinimumHeight(BTN_PRIMARY_WIDE_PX)
            retry.setCursor(Qt.CursorShape.PointingHandCursor)
            retry.setAutoDefault(False)
            retry.clicked.connect(lambda: self._lc_action("retry"))
            col.addWidget(retry)
            leave = QPushButton(tr("Exit"))
            leave.setStyleSheet(_BTN_AUTO_QUIET)
            leave.setCursor(Qt.CursorShape.PointingHandCursor)
            leave.setAutoDefault(False)
            leave.clicked.connect(lambda: self._lc_action("exit"))
            col.addWidget(leave, 0, Qt.AlignmentFlag.AlignHCenter)
            self._lc_error_box = box
            self._auto_review_column_layout.insertWidget(0, box)
        self.set_auto_review_active(True)
        for widget in self._land_cover_review_parts():
            widget.setVisible(False)
        panel = self.__dict__.get("auto_land_cover_panel")
        if panel is not None:
            panel.setVisible(False)
        box.setVisible(True)

    def set_land_cover_result(self, rows: list, total_m2: float,
                              min_patch_m2: float | None = None,
                              busy: bool = False) -> None:


        self._build_land_cover_card()
        from ...core.land_cover import land_cover_summary, whole_percent_shares

        ordered = sorted(rows, key=lambda r: (-float(r.get("area_m2") or 0.0), r["id"]))
        shares = whole_percent_shares([float(r.get("area_m2") or 0.0) for r in ordered])
        self._lc_total_label.setText(land_cover_area_text(total_m2))
        summary = land_cover_summary(ordered)
        if summary is None:
            self._lc_summary.setVisible(False)
        else:
            self._lc_summary.setText(
                tr("Impervious {a}% · Green {b}% · Water {c}%").format(
                    a=summary[0], b=summary[1], c=summary[2]))
            self._lc_summary.setVisible(True)
        self._lc_bar.set_parts([(r["color"], float(r.get("area_m2") or 0.0))
                                for r in ordered if r.get("shown", True)])
        while len(self._lc_rows) < len(ordered):
            row = _ClassRow(lambda cid: self._lc_action("hover", cid),
                            lambda cid: self._lc_action("toggle", cid))
            self._lc_rows_box.addWidget(row)
            self._lc_rows.append(row)


        metrics = QFontMetrics(self._lc_rows[0].area.font()) if self._lc_rows else None
        area_w = (max(metrics.horizontalAdvance(land_cover_area_text(total_m2)),
                      metrics.horizontalAdvance(land_cover_area_text(9999))) + 4
                  if metrics else 0)
        share_w = (metrics.horizontalAdvance(land_cover_share_text(100)) + 4
                   if metrics else 0)
        for i, row in enumerate(self._lc_rows):
            if i >= len(ordered):
                row.setVisible(False)
                continue
            data = ordered[i]
            area = float(data.get("area_m2") or 0.0)
            empty = area <= 0
            row.class_id = int(data["id"])
            row.name.setText(data["name"])
            row.name.setToolTip(data["name"])
            row.area.setText(land_cover_area_text(area))
            row.share.setText(land_cover_share_text(shares[i]))
            row.area.setMinimumWidth(area_w)
            row.share.setMinimumWidth(share_w)
            row.swatch.set_state(data["color"], bool(data.get("shown", True)), empty)
            row.eye.set_shown(bool(data.get("shown", True)))
            for label in (row.name, row.area, row.share):
                if label.property("empty") != empty:
                    label.setProperty("empty", empty)
                    label.style().unpolish(label)
                    label.style().polish(label)
            row.setVisible(True)
        if min_patch_m2 is not None:
            self._lc_patch_spin.blockSignals(True)
            self._lc_patch_spin.setValue(float(min_patch_m2))
            self._lc_patch_spin.blockSignals(False)
        self._lc_has_export_rows = any(
            r.get("shown", True) and float(r.get("area_m2") or 0.0) > 0 for r in rows)
        self.set_land_cover_busy(busy)

    def set_land_cover_export_success(self, classes: int, patches: int,
                                      layer_name: str, layer_id: str | None) -> None:


        try:
            from html import escape

            from .auto_recap import layer_link_html
            from .styles import msg_rich
            lbl = getattr(self, "auto_export_success", None)
            if lbl is None:
                return
            self._auto_recap_layer_id = layer_id or ""
            detail = tr("{c} classes, {n} patches, in {layer}").format(
                c=int(classes), n=grouped_locale().toString(int(patches)),
                layer=layer_link_html(layer_name, bool(layer_id)))
            lbl.setText(msg_rich("success", (
                f"<b>{escape(tr('Land cover saved'))}</b><br>"
                f'<span style="color: {INK_2};">{detail}</span>'), is_html=True))
            lbl.setToolTip(tr("Click the layer name to see it on the map") if layer_id else "")
            lbl.setVisible(True)
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    def set_land_cover_busy(self, busy: bool) -> None:
        host = self.__dict__.get("_lc_rows_host")
        if host is None:
            return
        host.setEnabled(not busy)
        host.setToolTip(tr("Updating") if busy else "")
        has_rows = self.__dict__.get("_lc_has_export_rows", False)
        self._lc_export_btn.setEnabled(not busy and has_rows)
        self._lc_export_btn.setToolTip(
            tr("Export only the classes shown on the map.") if has_rows
            else tr("Show a class on the map to export it."))
